package ai.robotbrain.face;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.media.audiofx.LoudnessEnhancer;
import android.os.SystemClock;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/** One bounded PCM stream; network credentials and robot actions never enter here. */
final class LiveAudio {
    interface Meter { void update(float level, boolean active); }
    private final Meter meter;
    private final ArrayBlockingQueue<byte[]> queue = new ArrayBlockingQueue<>(12);
    private volatile boolean active;
    private Thread thread;
    private volatile AudioTrack track;
    private volatile long written;
    private volatile long lastAppend;
    private String id = "";
    private String state = "idle";
    private int sequence;
    private byte[] lastChunk;
    LiveAudio(Meter meter) { this.meter=meter; }
    boolean active() { return active; }

    synchronized String start(String next) {
        if (next == null || !next.matches("live-[a-f0-9]{32}") || active || (thread!=null && thread.isAlive()) || next.equals(id)) return null;
        id=next; sequence=0; written=0; state="playing"; lastChunk=null;
        queue.clear(); active=true; lastAppend=SystemClock.elapsedRealtime();
        thread=new Thread(this::play,"robotbrain-live-pcm");
        thread.setDaemon(true); thread.start();
        return status(next);
    }
    synchronized String append(String target, int seq, byte[] pcm) {
        if (!active || !id.equals(target) || pcm.length==0 || pcm.length>6400 || (pcm.length&1)!=0) return null;
        if (seq == sequence-1 && java.util.Arrays.equals(lastChunk,pcm)) return status(target);
        if (seq != sequence || !queue.offer(pcm)) return null;
        sequence++; lastChunk=pcm; lastAppend=SystemClock.elapsedRealtime();
        return status(target);
    }
    synchronized String stop(String target) {
        if (!id.equals(target)) return null;
        active=false; state="stopped"; queue.clear();
        AudioTrack current=track;
        if (current!=null) try { current.pause(); current.flush(); } catch (RuntimeException ignored) { }
        meter.update(0,false);
        return status(target);
    }
    synchronized void shutdown() {
        active=false; state="stopped"; queue.clear();
        AudioTrack current=track;
        if (current!=null) try { current.pause(); current.flush(); } catch (RuntimeException ignored) { }
        meter.update(0,false);
    }
    synchronized String status(String target) {
        if (!id.equals(target)) return null;
        if (active) lastAppend=SystemClock.elapsedRealtime();
        long played=0;
        if (track!=null) try { played=track.getPlaybackHeadPosition()&0xffffffffL; } catch (RuntimeException ignored) { }
        return "{\"ok\":true,\"state\":\""+state+"\",\"sequence\":"+sequence+
            ",\"written_frames\":"+written+",\"played_frames\":"+played+",\"queued_chunks\":"+queue.size()+"}";
    }
    private void play() {
        AudioTrack player=null;
        LoudnessEnhancer louder=null;
        try {
            player=new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(16000)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(Math.max(6400,AudioTrack.getMinBufferSize(16000,AudioFormat.CHANNEL_OUT_MONO,AudioFormat.ENCODING_PCM_16BIT)))
                .build();
            try {
                louder=new LoudnessEnhancer(player.getAudioSessionId());
                louder.setTargetGain(AudioIO.SPEECH_BOOST_MB);
                louder.setEnabled(true);
            } catch (RuntimeException ignored) { louder=null; }
            track=player; player.play();
            while (active && SystemClock.elapsedRealtime()-lastAppend<30000) {
                byte[] pcm=queue.poll(200,TimeUnit.MILLISECONDS);
                if (pcm==null) { meter.update(0,false); continue; }
                long energy=0;
                for(int i=0;i<pcm.length;i+=2) { int v=(short)((pcm[i]&255)|(pcm[i+1]<<8)); energy+=(long)v*v; }
                float level=(float)Math.min(1,Math.sqrt(energy/(double)(pcm.length/2))/5000.0);
                meter.update(level,level>0.005);
                int offset=0;
                while(active && offset<pcm.length) {
                    int n=player.write(pcm,offset,pcm.length-offset,AudioTrack.WRITE_BLOCKING);
                    if(n<=0) throw new IllegalStateException("pcm_write_failed");
                    offset+=n; written+=n/2;
                }
            }
        } catch(Exception error) { synchronized(this) {state="failed";} }
        finally {
            if(louder!=null) try { louder.release(); } catch(RuntimeException ignored){}
            if(player!=null) {try {player.pause();player.flush();} catch(RuntimeException ignored){} player.release();}
            synchronized(this) {track=null;active=false;queue.clear();if(state.equals("playing"))state="expired";}
            meter.update(0,false);
        }
    }
}
