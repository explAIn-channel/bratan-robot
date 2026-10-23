/* RobotBrain body: one owner of both UARTs, one command stream, no sensor-age stops.
 * Protocol facts: VacuumTiger/sangam-io CRL200S documentation (see SOURCES.md).
 * Build: cc -O2 -std=c11 -Wall -Wextra -Werror body.c -lm -o body
 * --inhibit-motion is immutable for the entire process, including reconnects.
 */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define PI 3.14159265358979323846
#define CAP 16384
typedef struct { unsigned char bytes[CAP]; size_t n; uint64_t bad; } Buffer;
typedef struct {
    double tick_m, left_scale, right_scale, track, gyro_scale;
    double linear_scale, angular_scale, heading_gain, angular_limit, linear_accel, angular_accel, command_timeout;
    double pivot_kp, pivot_ki, pivot_effort_limit;
} Calibration;
static Calibration cal = {.00022230739747029513, .99761, 1.00239, .08213,
    .00017648, 4316.190456684954, 521.827426213211, 1.8, .6, .4, .8, .75, 0., 0., .6};
typedef struct {
    uint64_t seq, command_seq, bumper_events, touch_events, scan_seq, nonzero_tx;
    uint16_t left, right, start_button, dock_button;
    unsigned char raw[96];
    int have_body, bumper, ir, charging, docked, inhibit, lidar_pwm, hold, status_payload_bytes;
    double stamp, x, y, yaw, distance, v, w, gyro_bias, gyro_w, encoder_w, last_gap, max_gap, gyro_rest_s;
    double requested_v, requested_w, heading, sent_v, sent_w, voltage;
    double scan_stamp, scan_start, ranges[360], published[360], raw_last_angle;
    int scan_points, have_angle, scan_pending;
    unsigned char quality[360], published_quality[360];
    uint64_t lidar_packets, lidar_checksum_matches, lidar_checksum_mismatches;
    uint64_t heartbeats; double max_tick_gap;
} State;
static State state;
static double motion_not_before;
static int motion_blocked_at(double wall) { return state.inhibit || wall<motion_not_before; }
static int motion_inhibited(void) { return motion_blocked_at((double)time(NULL)); }
static volatile sig_atomic_t running = 1;
static int gd = -1, client = -1, authenticated, nav_mode, self_testing, laser_power = -1;
static double last_command_at;
static int board_led_command = -1;
static unsigned long long board_led_tx;
static double last_dock_contact = -1;
static int charging_led(int charging, int docked, double stamp, double *last_contact) {
    if (charging || docked) *last_contact=stamp;
    return *last_contact>=0 && stamp-*last_contact<5 ? 0 : 11;
}
static double mode_since, last_heartbeat;
static double pivot_integral, pivot_target;
static Buffer gd_buffer, laser_buffer;
static char token[129], boot_id[64];
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double clamp(double x, double lo, double hi) { return fmax(lo, fmin(hi, x)); }
static double wrap(double x) { return atan2(sin(x), cos(x)); }
static double pivot_feedback(double v, double w, double stamp, double dt) {
    /* Compensate slow wheel motion using measured yaw at the native 50 Hz loop.
     * Stale feedback reverts to the existing open-loop command, never extrapolates. */
    if (fabs(v)>.02 || !w || !state.have_body || stamp-state.stamp<0 ||
            stamp-state.stamp>.1 || motion_inhibited() || w*pivot_target<0) {
        pivot_integral=0;
        pivot_target=w;
        return w;
    }
    if (fabs(w)<fabs(pivot_target)) pivot_integral*=fabs(w/pivot_target);
    pivot_target=w;
    double error=w-state.w;
    double limit=cal.pivot_effort_limit;
    double next=clamp(pivot_integral+cal.pivot_ki*error*dt,-limit,limit);
    double output=w+cal.pivot_kp*error+next;
    if (fabs(output)<=limit || output*error<0) pivot_integral=next;
    /* A smaller target brakes by coasting; it must not reverse the wheels. */
    return clamp(w+cal.pivot_kp*error+pivot_integral,
                 w>0 ? 0 : -limit, w>0 ? limit : 0);
}
static double pivot_slew(double target, double previous, double dt) {
    /* Limit acceleration, not braking: accumulated effort must drop as soon
     * as gyro feedback detects overspeed or the phone reduces its target. */
    if (target*previous<0) return 0;
    if (fabs(target)<fabs(previous)) return target;
    return clamp(target,previous-cal.angular_accel*dt,previous+cal.angular_accel*dt);
}
static double heading_pivot(double limit, double error) {
    return fabs(error)<.015 ? 0 : clamp(cal.heading_gain*error,-fabs(limit),fabs(limit));
}
static uint16_t le16(const unsigned char *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint16_t be16(const unsigned char *p) { return (uint16_t)(p[0] << 8 | p[1]); }
static void put32(unsigned char *p, int32_t v) {
    uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; ++i) p[i] = (unsigned char)(u >> (8*i));
}
static uint16_t checksum(const unsigned char *p, size_t n) {
    uint16_t sum = 0; size_t i = 0;
    for (; i + 1 < n; i += 2) sum = (uint16_t)(sum + be16(p+i));
    if (i < n) sum ^= p[i];
    return sum;
}
static size_t packet(unsigned char *out, unsigned char cmd,
                     const unsigned char *payload, size_t n) {
    assert(n <= 96);
    out[0]=0xfa; out[1]=0xfb; out[2]=(unsigned char)(n+3); out[3]=cmd;
    if (n) memcpy(out+4, payload, n);
    uint16_t sum = checksum(out+3, n+1);
    out[4+n]=(unsigned char)(sum>>8); out[5+n]=(unsigned char)sum;
    return n+6;
}
static int write_all(int fd, const void *data, size_t n) {
    const unsigned char *p = data;
    while (n) {
        ssize_t done = write(fd, p, n);
        if (done < 0 && errno == EINTR) continue;
        if (done < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd wait = {fd, POLLOUT, 0};
            if (poll(&wait, 1, 10) > 0) continue;
        }
        if (done <= 0) return -1;
        p += done; n -= (size_t)done;
    }
    return 0;
}
static void transmit(unsigned char cmd, const unsigned char *p, size_t n) {
    unsigned char data[102]; size_t len = packet(data, cmd, p, n);
    if (gd >= 0 && write_all(gd, data, len) < 0) running = 0;
}
static void byte_command(unsigned char cmd, unsigned char value) { transmit(cmd, &value, 1); }
static void velocity(double v, double w) {
    unsigned char data[8];
    /* The bench interlock is at the only physical velocity writer. */
    if (motion_inhibited()) v = w = 0;
    v = clamp(v, -.24, .24);
    double limit=fabs(v)<=.02 ? fmax(cal.angular_limit,cal.pivot_effort_limit) : cal.angular_limit;
    w = clamp(w, -limit, limit);
    /* Stock AuxCtrl uses mode 1 / 0x67 with two i32 wheel velocities.
     * Mode 2 / 0x66 vetoes forward travel on the repurposed cliff inputs. */
    int32_t iv = (int32_t)lround((v-w*cal.track*.5)*cal.linear_scale);
    int32_t iw = (int32_t)lround((v+w*cal.track*.5)*cal.linear_scale);
    if (iv || iw) state.nonzero_tx++;
    put32(data, iv); put32(data+4, iw); transmit(0x67, data, 8);
    state.sent_v=v; state.sent_w=w;
}
static void stop_command(void) {
    state.requested_v = state.requested_w = 0; state.hold = 0;
    pivot_integral=pivot_target=0;
    velocity(0, 0);
}
static void shutdown_client(void) {
    if (client >= 0) close(client);
    client = -1; authenticated = 0; state.command_seq=0;
    state.lidar_pwm=0;
    stop_command();
}
static void body_events(const unsigned char *p) {
    int bumper=p[1]&6;
    if (bumper && !state.bumper) state.bumper_events++;
    uint16_t start=le16(p+0x3a), dock=le16(p+0x3e);
    if ((start && !state.start_button) || (dock && !state.dock_button)) state.touch_events++;
    state.bumper=bumper; state.start_button=start; state.dock_button=dock;
}
static void body_sample(const unsigned char *p, double stamp) {
    uint16_t left=le16(p+0x10), right=le16(p+0x18);
    body_events(p);
    double dt = stamp-state.stamp;
    if (state.have_body) {
        int dl=(int16_t)(left-state.left), dr=(int16_t)(right-state.right);
        double dl_m=dl*cal.tick_m*cal.left_scale, dr_m=dr*cal.tick_m*cal.right_scale;
        double ds=(dl_m+dr_m)*.5, enc_yaw=(dr_m-dl_m)/cal.track;
        double gyro_raw=(int16_t)le16(p+0x28);
        /* Zero commands and one tickless packet also occur while coasting.
         * Require a full second of fresh, uninterrupted rest before calibration. */
        if (dt>0 && dt<=.2 && !dl && !dr && fabs(state.sent_v)<1e-6 && fabs(state.sent_w)<1e-6 &&
            fabs(state.requested_v)<1e-6 && fabs(state.requested_w)<1e-6)
            state.gyro_rest_s+=dt;
        else state.gyro_rest_s=0;
        if (state.gyro_rest_s>=1.) {
            double a = state.seq < 100 ? 1.0/(double)(state.seq+1) : fmin(dt/8., .02);
            state.gyro_bias += a*(gyro_raw-state.gyro_bias);
        }
        double gyro_w=(gyro_raw-state.gyro_bias)*cal.gyro_scale;
        state.gyro_w=gyro_w; state.encoder_w=dt>0 ? enc_yaw/dt : 0;
        /* A single last gyro rate cannot reconstruct a long receive gap.
         * Encoder deltas still recover the travelled distance without a stop. */
        double dyaw=dt>0 && dt<=.2 ? .98*gyro_w*dt+.02*enc_yaw : enc_yaw;
        if (!dl && !dr && fabs(state.sent_v)<1e-6 && fabs(state.sent_w)<1e-6) dyaw=0;
        double midpoint=state.yaw+dyaw*.5;
        state.x += ds*cos(midpoint); state.y += ds*sin(midpoint);
        state.yaw=wrap(state.yaw+dyaw); state.distance += ds;
        if (dt > 0) {
            double a=dt/(.08+dt);
            state.v += a*(ds/dt-state.v); state.w += a*(dyaw/dt-state.w);
            state.last_gap=dt; state.max_gap=fmax(state.max_gap, dt);
        }
    } else if (!state.have_body) state.gyro_bias=(int16_t)le16(p+0x28);
    state.left=left; state.right=right; state.ir=p[3]&15;
    state.docked=!!(p[7]&1); state.charging=!!(p[7]&2); state.voltage=p[8]/10.;
    memcpy(state.raw,p,96); state.stamp=stamp; state.seq++; state.have_body=1;
}
static void consume(Buffer *b, size_t n) {
    memmove(b->bytes, b->bytes+n, b->n-n); b->n-=n;
}
static void feed_body(const unsigned char *p, size_t n, double stamp) {
    Buffer *b=&gd_buffer;
    unsigned char newest[96]; unsigned samples=0;
    if (n > CAP-b->n) { b->bad++; b->n=0; }
    if (n > CAP) return;
    memcpy(b->bytes+b->n,p,n); b->n+=n;
    while (b->n>=4) {
        if (b->bytes[0]!=0xfa || b->bytes[1]!=0xfb) { consume(b,1); b->bad++; continue; }
        size_t len=b->bytes[2], total=len+3;
        if (len<3 || len>200) { consume(b,1); b->bad++; continue; }
        if (b->n<total) break;
        if (checksum(b->bytes+3,len-2)!=be16(b->bytes+total-2)) {
            consume(b,1); b->bad++; continue;
        }
        if (b->bytes[3]==0x15) state.status_payload_bytes=(int)len-3;
        /* The highest consumed field is the u16 at offset 0x3e. Firmware
         * variants omit unused trailing bytes; validate the fields we read. */
        if (b->bytes[3]==0x15 && len>=67) {
            unsigned char sample[96]={0};
            memcpy(sample,b->bytes+4,len-3<96 ? len-3 : 96);
            if (!state.have_body) body_sample(sample,stamp);
            else {
                body_events(sample);
                memcpy(newest,sample,sizeof(newest)); samples++;
            }
        }
        consume(b,total);
    }
    /* Integrate the whole received encoder delta once. Coalesced packets have
     * one measured arrival time: treating every packet as another gyro period
     * double-counts rotation, while using only its last delta loses distance.
     * Button/bumper edges above are retained even between published samples. */
    if (samples) { body_sample(newest,stamp); state.seq+=samples-1; }
}
static void publish_scan(double stamp) {
    if (state.scan_points >= 50) {
        memcpy(state.published,state.ranges,sizeof(state.ranges));
        memcpy(state.published_quality,state.quality,sizeof(state.quality));
        state.scan_stamp=state.scan_start ? (state.scan_start+stamp)*.5 : stamp;
        state.scan_seq++; state.scan_pending=1;
    }
    memset(state.ranges,0,sizeof(state.ranges)); memset(state.quality,0,sizeof(state.quality)); state.scan_points=0; state.scan_start=stamp;
}
static void laser_sample(const unsigned char *p, size_t n, double stamp) {
    if (n<8 || (n-5)%3) return;
    /* Each Delta-2D packet spans 22.5 degrees. Bytes 1-2 are
     * a calibration offset, not sample spacing (confirmed on live packets). */
    double step=22.5/((n-5)/3), start=be16(p+3)*.01;
    if (start>360 || step<=0 || step>10) return;
    for (size_t i=5, index=0; i+2<n; i+=3,index++) {
        double raw_angle=start+index*step;
        if (raw_angle>360.01) continue;
        if (state.have_angle && raw_angle+180 < state.raw_last_angle) publish_scan(stamp);
        state.have_angle=1; state.raw_last_angle=raw_angle;
        double distance=be16(p+i+1)*.00025;
        if (!p[i] || distance<.2 || distance>8.) continue;
        /* Delta2D zero points backward and angles run clockwise. Reproduce the
         * donor LaserScan frame (-angle + pi); phone adds measured mount yaw. */
        int bin=(int)nearbyint(180.-raw_angle);
        bin=((bin%360)+360)%360;
        if (!state.ranges[bin] || distance<state.ranges[bin]) {
            state.ranges[bin]=distance; state.quality[bin]=p[i];
        }
        state.scan_points++;
    }
}
static void feed_laser(const unsigned char *p, size_t n, double stamp) {
    Buffer *b=&laser_buffer;
    if (n>CAP-b->n) { b->bad++; b->n=0; }
    if (n>CAP) return;
    memcpy(b->bytes+b->n,p,n); b->n+=n;
    while (b->n>=8) {
        size_t payload=be16(b->bytes+6), total=payload+10;
        if (b->bytes[0]!=0xaa || b->bytes[3]>15 || payload>1024 ||
            (b->bytes[5]!=0xad && b->bytes[5]!=0xae)) {
            consume(b,1); b->bad++; continue;
        }
        if (b->n<total) break;
        /* Sum-of-bytes checksum qualified against thousands of real packets. */
        uint16_t sum=0;
        for (size_t i=0;i<total-2;i++) sum=(uint16_t)(sum+b->bytes[i]);
        if (sum==be16(b->bytes+total-2)) state.lidar_checksum_matches++;
        else { state.lidar_checksum_mismatches++; consume(b,1); b->bad++; continue; }
        state.lidar_packets++;
        if (b->bytes[5]==0xad) laser_sample(b->bytes+8,payload,stamp);
        consume(b,total);
    }
}
static int equal_token(const char *given) {
    size_t n=strlen(token); unsigned diff=(unsigned)(strlen(given)^n);
    if (strlen(given)!=n) return 0;
    for (size_t i=0;i<n;i++) diff|=(unsigned char)given[i]^(unsigned char)token[i];
    return !diff;
}
static int command(const char *line) {
    if (!authenticated) {
        if (!strncmp(line,"AUTH ",5) && equal_token(line+5)) { authenticated=1; return 1; }
        return -1;
    }
    unsigned long long seq; double v,w,heading; int hold,pwm,used=0;
    if (sscanf(line,"SET %llu %lf %lf %d %lf %d%n",&seq,&v,&w,&hold,&heading,&pwm,&used)==6 &&
        !line[used] && isfinite(v) && isfinite(w) && isfinite(heading) &&
        fabs(v)<=.240001 && fabs(w)<=cal.angular_limit && (hold==0 || hold==1) && pwm>=0 && pwm<=100) {
        if (seq<=state.command_seq) return 0;
        state.command_seq=seq; state.requested_v=v; state.requested_w=w;
        state.hold=hold; state.heading=wrap(heading); state.lidar_pwm=pwm;
        last_command_at=now_s();
        if (!v && !w) stop_command();
        return 1;
    }
    used=0;
    if (sscanf(line,"STOP %llu%n",&seq,&used)==1 && !line[used]) {
        if (seq<=state.command_seq) return 0;
        state.command_seq=seq; last_command_at=now_s(); stop_command(); return 1;
    }
    return -1;
}
static void heartbeat(double stamp) {
    double dt=last_heartbeat ? clamp(stamp-last_heartbeat,0,.05) : .02;
    last_heartbeat=stamp;
    if ((state.requested_v || state.requested_w || state.lidar_pwm) && last_command_at &&
            stamp-last_command_at>cal.command_timeout) {
        /* Stop wheels immediately, but allow a short sensor-only grace period:
         * cycling lidar power on every delayed command destroys fresh scans.
         * Explicit off/disconnect still powers down immediately. */
        if (stamp-last_command_at>fmax(cal.command_timeout,5.)) state.lidar_pwm=0;
        stop_command();
    }
    int active=state.lidar_pwm || (!motion_inhibited() && (state.requested_v || state.requested_w));
    if (active && !nav_mode) { byte_command(0x65,1); nav_mode=1; mode_since=stamp; }
    if (!active && nav_mode) {
        velocity(0,0); unsigned char zero[4]={0}; transmit(0x71,zero,4);
        byte_command(0x97,0); laser_power=0; byte_command(0x65,0); nav_mode=0;
    }
    if (active) {
        byte_command(0x65,1);
        double v=state.requested_v, w=state.requested_w;
        if (state.hold && fabs(v)>0) w+=cal.heading_gain*wrap(state.heading-state.yaw);
        else if (state.hold) w=heading_pivot(w,wrap(state.heading-state.yaw));
        w=clamp(w,-cal.angular_limit,cal.angular_limit);
        if (stamp-mode_since<.1) v=w=0;
        w=pivot_feedback(v,w,stamp,dt);
        v=clamp(v,state.sent_v-cal.linear_accel*dt,state.sent_v+cal.linear_accel*dt);
        w=(fabs(v)<=.02 && (cal.pivot_kp || cal.pivot_ki)) ? pivot_slew(w,state.sent_w,dt) :
            clamp(w,state.sent_w-cal.angular_accel*dt,state.sent_w+cal.angular_accel*dt);
        velocity(v,w);
        unsigned char pwm[4]; put32(pwm,state.lidar_pwm); transmit(0x71,pwm,4);
        int power=!!state.lidar_pwm;
        if (power!=laser_power) { byte_command(0x97,(unsigned char)power); laser_power=power; }
    } else {
        transmit(0x06,NULL,0);
        if (state.heartbeats%10==0) {
            /* Reassert quiet actuators even after an MCU state change. */
            unsigned char zero[4]={0};
            transmit(0x71,zero,4); byte_command(0x97,0); byte_command(0x65,0);
            transmit(0x68,zero,2);
            byte_command(0x69,0); byte_command(0x6a,0); byte_command(0x6b,0);
        }
    }
    if (state.heartbeats%75==0) transmit(0x0d,NULL,0);
    if (state.have_body) {
        int led=charging_led(state.charging,state.docked,stamp,&last_dock_contact);
        if (led!=board_led_command || !led) {
            byte_command(0x8d,(unsigned char)led);
            board_led_command=led; board_led_tx++;
        }
    }
    state.heartbeats++;
}
static size_t body_json(char *out, size_t n) {
    int used=snprintf(out,n,
        "{\"type\":\"body\",\"boot\":\"%s\",\"seq\":%llu,\"board_s\":%.6f,"
        "\"sample_s\":%.6f,\"encoders\":[%u,%u],\"odom\":[%.7f,%.7f,%.7f],"
        "\"distance\":%.7f,\"twist\":[%.5f,%.5f],\"gyro_bias\":%.3f,\"gyro_w\":%.6f,\"encoder_w\":%.6f,\"linear_limit\":0.24,"
        "\"bumper\":%d,\"ir\":%d,\"bumper_events\":%llu,\"touch_events\":%llu,"
        "\"start_button\":%u,\"dock_button\":%u,\"docked\":%s,\"charging\":%s,"
        "\"board_led_command\":%d,\"board_led_tx\":%llu,\"voltage\":%.2f,\"inhibit_motion\":%s,\"motion_not_before\":%.0f,\"lidar_pwm\":%d,"
        "\"command_seq\":%llu,\"requested\":[%.5f,%.5f],\"sent\":[%.5f,%.5f],\"status_payload_bytes\":%d,"
        "\"sensor_gap_s\":%.6f,\"max_sensor_gap_s\":%.6f,\"max_tick_gap_s\":%.6f,"
        "\"rx_errors\":%llu,\"laser_errors\":%llu,\"nonzero_tx\":%llu,"
        "\"lidar_packets\":%llu,\"lidar_checksum_matches\":%llu,\"lidar_checksum_mismatches\":%llu,"
        "\"imu_raw\":[%d,%d,%d,%d,%d,%d],\"raw_hex\":\"",
        boot_id,(unsigned long long)state.seq,now_s(),state.stamp,state.left,state.right,
        state.x,state.y,state.yaw,state.distance,state.v,state.w,state.gyro_bias,state.gyro_w,state.encoder_w,
        state.bumper,state.ir,(unsigned long long)state.bumper_events,(unsigned long long)state.touch_events,
        state.start_button,state.dock_button,state.docked?"true":"false",state.charging?"true":"false",
        board_led_command,board_led_tx,state.voltage,motion_inhibited()?"true":"false",motion_not_before,state.lidar_pwm,(unsigned long long)state.command_seq,
        state.requested_v,state.requested_w,state.sent_v,state.sent_w,state.status_payload_bytes,state.last_gap,state.max_gap,state.max_tick_gap,
        (unsigned long long)gd_buffer.bad,(unsigned long long)laser_buffer.bad,(unsigned long long)state.nonzero_tx,
        (unsigned long long)state.lidar_packets,(unsigned long long)state.lidar_checksum_matches,
        (unsigned long long)state.lidar_checksum_mismatches,
        (int16_t)le16(state.raw+0x28),(int16_t)le16(state.raw+0x2c),(int16_t)le16(state.raw+0x30),
        (int16_t)le16(state.raw+0x2a),(int16_t)le16(state.raw+0x2e),(int16_t)le16(state.raw+0x32));
    if (used<0 || (size_t)used+198>=n) return 0;
    for (int i=0;i<96;i++) used+=snprintf(out+used,n-(size_t)used,"%02x",state.raw[i]);
    used+=snprintf(out+used,n-(size_t)used,"\"}\n");
    return (size_t)used;
}
static size_t scan_json(char *out,size_t n) {
    int used=snprintf(out,n,"{\"type\":\"scan\",\"boot\":\"%s\",\"seq\":%llu,"
        "\"board_s\":%.6f,\"range_min\":0.15,\"range_max\":12,\"ranges\":[",
        boot_id,(unsigned long long)state.scan_seq,state.scan_stamp);
    if (used<0) return 0;
    for (int i=0;i<360;i++) {
        if ((size_t)used+32>=n) return 0;
        used+=snprintf(out+used,n-(size_t)used,"%s%.4f",i?",":"",state.published[i]);
    }
    used+=snprintf(out+used,n-(size_t)used,"],\"qualities\":[");
    for (int i=0;i<360;i++) {
        if ((size_t)used+16>=n) return 0;
        used+=snprintf(out+used,n-(size_t)used,"%s%u",i?",":"",state.published_quality[i]);
    }
    used+=snprintf(out+used,n-(size_t)used,"]}\n"); return (size_t)used;
}
static int serial_port(const char *path) {
    int fd=open(path,O_RDWR|O_NOCTTY|O_NONBLOCK);
    if (fd<0) { perror("serial open"); return -1; }
    struct termios term;
    if (tcgetattr(fd,&term)) { close(fd); return -1; }
    cfmakeraw(&term); cfsetispeed(&term,B115200); cfsetospeed(&term,B115200);
    term.c_cflag|=CLOCAL|CREAD; term.c_cflag&=~CRTSCTS;
    if (tcsetattr(fd,TCSANOW,&term)) { close(fd); return -1; }
    tcflush(fd,TCIFLUSH); return fd;
}
static void signal_stop(int signum) { (void)signum; running=0; }
static int calibration(const char *path) {
    FILE *file=fopen(path,"r"); if (!file) return -1;
    char name[80]; double v;
    while (fscanf(file,"%79s %lf",name,&v)==2) {
        if (!isfinite(v)) { fclose(file); return -1; }
        if (!strcmp(name,"tick_m") && v>0 && v<.01) cal.tick_m=v;
        else if (!strcmp(name,"left_scale") && v>.5 && v<2) cal.left_scale=v;
        else if (!strcmp(name,"right_scale") && v>.5 && v<2) cal.right_scale=v;
        else if (!strcmp(name,"track") && v>.03 && v<.5) cal.track=v;
        else if (!strcmp(name,"gyro_scale") && fabs(v)<.01 && v!=0) cal.gyro_scale=v;
        else if (!strcmp(name,"linear_scale") && v>0 && v<100000) cal.linear_scale=v;
        else if (!strcmp(name,"angular_scale") && v>0 && v<100000) cal.angular_scale=v;
        else if (!strcmp(name,"heading_gain") && v>=0 && v<20) cal.heading_gain=v;
        else if (!strcmp(name,"angular_limit") && v>.01 && v<=1.) cal.angular_limit=v;
        else if (!strcmp(name,"linear_accel") && v>.01 && v<=2.) cal.linear_accel=v;
        else if (!strcmp(name,"angular_accel") && v>.01 && v<=5.) cal.angular_accel=v;
        else if (!strcmp(name,"command_timeout") && v>=.2 && v<=5.) cal.command_timeout=v;
        else if (!strcmp(name,"pivot_kp") && v>=0 && v<=3.) cal.pivot_kp=v;
        else if (!strcmp(name,"pivot_ki") && v>=0 && v<=5.) cal.pivot_ki=v;
        else if (!strcmp(name,"pivot_effort_limit") && v>=.1 && v<=1.) cal.pivot_effort_limit=v;
        else { fclose(file); return -1; }
    }
    int bad=!feof(file); fclose(file); return bad?-1:0;
}
static void self_test(void) {
    self_testing=1; unsigned char raw[96]={0}, packed[102];
    assert(pivot_slew(.1,.9,.02)==.1 && pivot_slew(-.1,-.9,.02)==-.1);
    assert(pivot_slew(-.3,.9,.02)==0 && pivot_slew(0,.9,.02)==0);
    assert(pivot_slew(.9,0,.02)<=cal.angular_accel*.02);
    assert(heading_pivot(.35,.01)==0 && heading_pivot(-.35,-.01)==0);
    assert(heading_pivot(.35,1.)==.35 && heading_pivot(-.35,-1.)==-.35);
    cal.pivot_kp=1.; cal.pivot_ki=1.5; cal.pivot_effort_limit=1.; state.have_body=1;
    for (int sign=-1;sign<=1;sign+=2) {
        state.w=0; pivot_integral=pivot_target=0;
        for (int i=0;i<500;i++) {
            state.stamp=10+i*.02;
            double output=pivot_feedback(0,sign*.25,state.stamp,.02);
            assert(fabs(output)<=cal.pivot_effort_limit && output*sign>=0);
            double physical=copysign(fmax(0.,fabs(output)-.22)*.85,output);
            state.w+=(physical-state.w)*.1;
        }
        assert(fabs(state.w-sign*.25)<.01);
        assert(pivot_feedback(0,0,state.stamp,.02)==0 && pivot_integral==0);
        assert(pivot_feedback(.16,.2,state.stamp,.02)==.2);
        assert(pivot_feedback(0,.2,state.stamp+.2,.02)==.2);
    }
    state=(State){0}; pivot_integral=pivot_target=0; cal.pivot_kp=cal.pivot_ki=0; cal.pivot_effort_limit=.6;
    unsigned char heartbeat_expected[]={0xfa,0xfb,3,6,0,6}, mode[]={2};
    double led_contact=-1;
    assert(charging_led(0,0,10,&led_contact)==11);
    assert(charging_led(1,0,12,&led_contact)==0 && charging_led(0,0,13,&led_contact)==0);
    assert(charging_led(0,0,18,&led_contact)==11 && charging_led(0,1,19,&led_contact)==0);
    assert(packet(packed,6,NULL,0)==6 && !memcmp(packed,heartbeat_expected,6));
    packet(packed,0x65,mode,1); assert(packed[5]==0x65 && packed[6]==2);
    raw[0x10]=0xfe; raw[0x11]=0xff; raw[0x18]=0xfe; raw[0x19]=0xff; raw[7]=3; raw[8]=156;
    packet(packed,0x15,raw,96);
    feed_body(packed,17,1.); assert(!state.have_body);
    feed_body(packed+17,85,1.); assert(state.seq==1 && state.charging && state.docked);
    raw[0x10]=raw[0x18]=2; raw[0x11]=raw[0x19]=0; raw[1]=2;
    packet(packed,0x15,raw,96); feed_body(packed,102,1.01);
    assert(state.seq==2 && state.distance>.0008 && state.distance<.001);
    assert(state.bumper_events==1);
    double distance_before=state.distance;
    raw[0x10]=raw[0x18]=4; packet(packed,0x15,raw,96);
    feed_body(packed,102,1.01); /* Coalesced serial reads may share a timestamp. */
    assert(state.distance>distance_before+.0004);
    packed[30]^=1; feed_body(packed,102,1.02); assert(state.seq==3 && gd_buffer.bad>0);
    double yaw_before=state.yaw;
    raw[0x28]=0xe8; raw[0x29]=3; state.sent_w=.1;
    body_sample(raw,1.11);
    assert(state.yaw>yaw_before+.01); /* Physical left turn has positive raw yaw. */
    /* Braking samples with no encoder tick must not become the gyro zero. */
    state=(State){0}; memset(raw,0,sizeof(raw)); body_sample(raw,10.);
    for (int i=0;i<200;i++) {
        state.sent_w=.2; raw[0x28]=0x18; raw[0x29]=0xfc; /* -1000 */
        body_sample(raw,10.02+i*.06);
        state.sent_w=0; body_sample(raw,10.04+i*.06);
        raw[0x28]=raw[0x29]=0; body_sample(raw,10.06+i*.06);
    }
    assert(fabs(state.gyro_bias)<1e-9);
    /* A long receive gap is not evidence of continuous rest. */
    raw[0x28]=40; body_sample(raw,30.);
    assert(fabs(state.gyro_bias)<1e-9);
    for (int i=1;i<=1000;i++) body_sample(raw,30.+i*.02);
    assert(state.gyro_bias>35 && state.gyro_bias<40);
    strcpy(token,"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    assert(command("AUTH wrong")==-1);
    char auth[100]; snprintf(auth,sizeof(auth),"AUTH %s",token); assert(command(auth)==1);
    assert(command("SET 1 .16 0 1 0 60")==1);
    double command_at=last_command_at;
    assert(command("SET 1 -.16 0 0 0 0")==0 && state.requested_v==.16);
    assert(command("SET 2 nan 0 0 0 0")==-1);
    assert(command("SET 2 .241 0 0 0 0")==-1);
    assert(command("SET 2 .1 0 0 0 0 trailing")==-1);
    heartbeat(command_at+cal.command_timeout+.01);
    assert(state.requested_v==0 && state.sent_v==0);
    assert(state.lidar_pwm==60); /* Wheel watchdog must not repeatedly reboot scanning. */
    heartbeat(command_at+5.01); assert(state.lidar_pwm==0);
    assert(command("SET 2 0 0 0 0 60")==1);
    heartbeat(last_command_at+cal.command_timeout+.01);
    assert(state.lidar_pwm==60);
    heartbeat(last_command_at+5.01);
    assert(state.lidar_pwm==0); /* A stalled phone must not drain a parked robot. */
    state.lidar_pwm=60;
    shutdown_client();
    assert(state.lidar_pwm==0 && !state.requested_v && !state.requested_w);
    assert(command(auth)==1);
    motion_not_before=(double)time(NULL)+3600;
    velocity(.16,.2); assert(!state.nonzero_tx && !state.sent_v && !state.sent_w);
    assert(motion_blocked_at(motion_not_before-1) && !motion_blocked_at(motion_not_before));
    state.inhibit=1; velocity(.16,.2); assert(!state.nonzero_tx && state.sent_v==0 && state.sent_w==0);
    assert(motion_blocked_at(motion_not_before+1));
    motion_not_before=0;
    assert(command("STOP 3")==1 && state.requested_v==0);
    char json[CAP]; assert(body_json(json,sizeof(json))>0);
    state.scan_points=51; state.ranges[0]=1.25; publish_scan(2.);
    assert(state.scan_seq==1 && state.published[0]==1.25 && !state.ranges[0]);
    assert(scan_json(json,sizeof(json))>0);
    /* Real packet shape: 25 samples per 22.5-degree sector; offset .56
     * differs from the .9-degree sample spacing. Also exercise scan wrap. */
    unsigned char laser[90]={0xaa,0,88,1,0x61,0xad,0,80,0x66,0,56,0,0};
    for (size_t i=13;i<88;i+=3) { laser[i]=100; laser[i+1]=0x0f; laser[i+2]=0xa0; }
    state.have_angle=0;
    for (unsigned sector=0;sector<=16;sector++) {
        unsigned angle=(sector%16)*2250, sum=0;
        laser[11]=(unsigned char)(angle>>8); laser[12]=(unsigned char)angle;
        for (size_t i=0;i<88;i++) sum+=laser[i];
        laser[88]=(unsigned char)(sum>>8); laser[89]=(unsigned char)sum;
        feed_laser(laser,37,3.+sector*.01);
        feed_laser(laser+37,sizeof(laser)-37,3.+sector*.01);
    }
    assert(state.scan_seq==2 && state.published[180]==1. && state.published[158]==1.);
    assert(state.published[90]==1. && state.published[0]==1.);
    assert(state.published_quality[158]==100 && state.published_quality[90]==100);
    assert(scan_json(json,sizeof(json))>0 && strstr(json,"qualities"));
    laser[14]^=1; feed_laser(laser,sizeof(laser),3.4);
    assert(state.lidar_checksum_mismatches==1 && state.scan_seq==2);
    puts("body self-test: packets, corruption, wrap, command ordering, immutable inhibit PASS");
}
int main(int argc,char **argv) {
    const char *gd_path="/dev/ttyS3", *laser_path="/dev/ttyS1", *key_path=NULL;
    const char *lock_path="/tmp/robotbrain-body.lock"; int port=8800;
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--self-test")) { self_test(); return 0; }
        if (!strcmp(argv[i],"--inhibit-motion")) { state.inhibit=1; continue; }
        if (i+1>=argc) { fprintf(stderr,"missing option value\n"); return 2; }
        if (!strcmp(argv[i],"--serial")) gd_path=argv[++i];
        else if (!strcmp(argv[i],"--lidar")) laser_path=argv[++i];
        else if (!strcmp(argv[i],"--key")) key_path=argv[++i];
        else if (!strcmp(argv[i],"--lock")) lock_path=argv[++i];
        else if (!strcmp(argv[i],"--port")) port=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--motion-not-before")) {
            const char *value=argv[++i];
            if (!*value || strspn(value,"0123456789")!=strlen(value)) return 2;
            motion_not_before=strtod(value,NULL);
            if (!isfinite(motion_not_before) || motion_not_before>4102444800.) return 2;
        }
        else if (!strcmp(argv[i],"--calibration")) {
            if (calibration(argv[++i])) { fprintf(stderr,"invalid calibration\n"); return 2; }
        } else { fprintf(stderr,"unknown option\n"); return 2; }
    }
    if (!key_path || port<0 || port>65535) { fprintf(stderr,"--key is required\n"); return 2; }
    FILE *key=fopen(key_path,"r");
    if (!key || !fgets(token,sizeof(token),key)) { fprintf(stderr,"key unavailable\n"); return 2; }
    fclose(key); token[strcspn(token,"\r\n")]=0;
    if (strlen(token)!=64 || strspn(token,"0123456789abcdef")!=64) { fprintf(stderr,"invalid key\n"); return 2; }
    int lock_fd=open(lock_path,O_CREAT|O_RDWR,0600);
    if (lock_fd<0 || flock(lock_fd,LOCK_EX|LOCK_NB)) { fprintf(stderr,"body already running\n"); return 2; }
    gd=serial_port(gd_path); if (gd<0) return 2;
    int laser=!strcmp(laser_path,"-")?-1:serial_port(laser_path);
    if (laser<0 && strcmp(laser_path,"-")) { close(gd); return 2; }
    int server=socket(AF_INET,SOCK_STREAM,0), one=1;
    setsockopt(server,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    struct sockaddr_in addr={0}; addr.sin_family=AF_INET; addr.sin_port=htons((uint16_t)port); addr.sin_addr.s_addr=INADDR_ANY;
    if (server<0 || bind(server,(struct sockaddr*)&addr,sizeof(addr)) || listen(server,2)) { perror("listen"); return 2; }
    fcntl(server,F_SETFL,O_NONBLOCK);
    socklen_t addr_len=sizeof(addr); getsockname(server,(struct sockaddr*)&addr,&addr_len);
    snprintf(boot_id,sizeof(boot_id),"%ld-%.0f",(long)getpid(),now_s()*1e6);
    signal(SIGINT,signal_stop); signal(SIGTERM,signal_stop); signal(SIGPIPE,SIG_IGN);
    fprintf(stderr,"body ready port=%u inhibit_motion=%d\n",ntohs(addr.sin_port),motion_inhibited());
    unsigned char init[]={0xfa,0xfb,1,8};
    write_all(gd,init,sizeof(init)); stop_command(); byte_command(0x65,0); byte_command(0x78,1);
    /* No vacuum, brush, water or power-management commands are ever issued. */
    char input[2048], output[CAP]; size_t input_n=0, output_n=0, output_offset=0;
    double next_tick=now_s(), last_tick=next_tick, next_body=next_tick;
    while (running) {
        struct pollfd fds[4]={{server,POLLIN,0},{gd,POLLIN,0},{laser,POLLIN,0},{client,POLLIN,0}};
        if (client>=0 && output_n>output_offset) fds[3].events|=POLLOUT;
        double stamp=now_s(); int wait_ms=(int)clamp((next_tick-stamp)*1000,0,20);
        int ready=poll(fds,4,wait_ms);
        if (ready<0 && errno!=EINTR) break;
        stamp=now_s();
        if (fds[0].revents&POLLIN) {
            int incoming=accept(server,NULL,NULL);
            if (incoming>=0) {
                if (client>=0) close(incoming);
                else {
                    client=incoming; fcntl(client,F_SETFL,O_NONBLOCK);
                    setsockopt(client,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
                    authenticated=0; input_n=output_n=output_offset=0;
                }
            }
        }
        for (int i=1;i<=2;i++) if (fds[i].revents&POLLIN) {
            unsigned char bytes[4096]; ssize_t got;
            while ((got=read(fds[i].fd,bytes,sizeof(bytes)))>0) {
                if (i==1) feed_body(bytes,(size_t)got,stamp); else feed_laser(bytes,(size_t)got,stamp);
            }
        }
        if (client>=0 && fds[3].fd==client && fds[3].revents&(POLLHUP|POLLERR|POLLNVAL)) shutdown_client();
        if (client>=0 && fds[3].fd==client && fds[3].revents&POLLIN) {
            ssize_t got=read(client,input+input_n,sizeof(input)-1-input_n);
            if (got<=0) shutdown_client();
            else {
                input_n+=(size_t)got; input[input_n]=0;
                char *line=input, *end;
                while ((end=strchr(line,'\n'))) {
                    *end=0; if (end>line && end[-1]=='\r') end[-1]=0;
                    if (command(line)<0) { shutdown_client(); break; }
                    line=end+1;
                }
                if (client>=0) {
                    size_t remaining=input_n-(size_t)(line-input); memmove(input,line,remaining); input_n=remaining;
                    if (input_n==sizeof(input)-1) shutdown_client();
                }
            }
        }
        if (stamp>=next_tick) {
            state.max_tick_gap=fmax(state.max_tick_gap,stamp-last_tick); last_tick=stamp;
            heartbeat(stamp); next_tick+=.02;
            if (next_tick<stamp) next_tick=stamp+.02;
        }
        if (client>=0 && authenticated) {
            if (output_offset==output_n) {
                output_n=output_offset=0;
                if (state.scan_pending) { output_n=scan_json(output,sizeof(output)); state.scan_pending=0; }
                else if (stamp>=next_body) { output_n=body_json(output,sizeof(output)); next_body=stamp+.02; }
            }
            if (output_n>output_offset) {
                ssize_t sent=send(client,output+output_offset,output_n-output_offset,MSG_DONTWAIT|MSG_NOSIGNAL);
                if (sent>0) output_offset+=(size_t)sent;
                else if (sent<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) shutdown_client();
            }
        }
    }
    stop_command(); unsigned char zero[4]={0}; transmit(0x71,zero,4); byte_command(0x97,0); byte_command(0x65,0);
    if (client>=0) close(client);
    close(server); close(gd); if (laser>=0) close(laser); close(lock_fd);
    fprintf(stderr,"body stopped nonzero_tx=%llu\n",(unsigned long long)state.nonzero_tx);
    return self_testing?1:0;
}
