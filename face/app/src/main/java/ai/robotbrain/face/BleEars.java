package ai.robotbrain.face;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.ParcelUuid;
import android.os.SystemClock;
import android.util.Log;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.Collections;
import java.util.UUID;

/**
 * The body's ears over Bluetooth LE (18.10, the owner: «подружи телефон с платой напрямую — они всегда рядом»).
 *
 * The ESP32 of the ears sits at the bottom of the body and its Wi-Fi fell to −85…−93 dBm (70 % lost); the phone is
 * 30 cm away. The board sends the same BRM1 packets (three microphones, IMA-ADPCM, 32 ms each) as notifications,
 * cut to the MTU: every chunk starts with a byte — bit 7 «a packet starts here» (then its length, u16 LE), bits 0-6 a
 * counter, so a lost chunk drops one packet, not the stream. Whole packets go to the robot on 127.0.0.1:5772, where
 * voice/array.py takes them exactly like the ones that come by Wi-Fi.
 */
@SuppressLint("MissingPermission")
final class BleEars {
    static final UUID SERVICE = UUID.fromString("6b9c0001-3f6e-4f4b-9a54-6272617461e1");
    static final UUID AUDIO = UUID.fromString("6b9c0002-3f6e-4f4b-9a54-6272617461e1");
    // The board streams only to the connection that wrote its token here (anyone in range could listen otherwise).
    static final UUID UNLOCK = UUID.fromString("6b9c0003-3f6e-4f4b-9a54-6272617461e1");
    private static final UUID CCCD = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");
    private static final int LOCAL_PORT = 5772;
    private static final long SCAN_MS = 10_000L;
    private static final long QUIET_MS = 3_000L;         // nothing for this long while connected: start over

    private final Context context;
    private final BluetoothAdapter adapter;
    private final Handler handler;
    private DatagramSocket socket;
    private InetAddress loopback;
    private BluetoothGatt gatt;
    private String address = "";
    private boolean scanning, stopping;
    private String status = "starting";
    private final byte[] packet = new byte[4096];
    private int expected = -1, filled = 0, lastCounter = -1;
    private long packets, chunks, dropped, lastDataMs;
    private int mtu = 23;

    BleEars(Context context) {
        this.context = context.getApplicationContext();
        BluetoothManager manager = (BluetoothManager)context.getSystemService(Context.BLUETOOTH_SERVICE);
        adapter = manager == null ? null : manager.getAdapter();
        HandlerThread thread = new HandlerThread("ble-ears");
        thread.start();
        handler = new Handler(thread.getLooper());
    }

    private boolean started;

    synchronized void start() {
        if (started) return;                           // onCreate and the permission answer both call it
        started = true;
        stopping = false;
        handler.post(this::beginScan);
        handler.postDelayed(watch, QUIET_MS);
    }

    synchronized void stop() {
        stopping = true;
        stopScan();
        if (gatt != null) {
            try {
                gatt.disconnect();                   // the link down too, not only this app's use of it
            } catch (RuntimeException ignored) {
            }
            gatt.close();
            gatt = null;
        }
    }

    synchronized String statusJson() {
        long age = lastDataMs == 0 ? -1 : SystemClock.elapsedRealtime() - lastDataMs;
        return "{\"status\":\"" + status + "\",\"address\":\"" + address + "\",\"mtu\":" + mtu + ",\"packets\":" + packets +
                ",\"chunks\":" + chunks + ",\"dropped\":" + dropped + ",\"data_age_ms\":" + age + "}";
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= 31) {
            return context.checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
                    && context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) ==
                    PackageManager.PERMISSION_GRANTED;
        }
        return context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED;
    }

    private synchronized void beginScan() {
        if (stopping || scanning || gatt != null) return;
        if (adapter == null) { status = "bluetooth_unavailable"; return; }
        if (!hasPermission()) { status = "permission_missing"; handler.postDelayed(this::beginScan, 5000L); return; }
        if (!adapter.isEnabled()) { status = "bluetooth_off"; handler.postDelayed(this::beginScan, 2000L); return; }
        if (!address.isEmpty()) {                      // known board: straight to it, no scan (Android limits scans)
            connect(adapter.getRemoteDevice(address));
            return;
        }
        BluetoothLeScanner scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) { status = "scanner_unavailable"; handler.postDelayed(this::beginScan, 3000L); return; }
        scanning = true;
        status = "scanning";
        ScanFilter filter = new ScanFilter.Builder().setServiceUuid(new ParcelUuid(SERVICE)).build();
        ScanSettings settings = new ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build();
        scanner.startScan(Collections.singletonList(filter), settings, scanCallback);
        handler.postDelayed(scanTimeout, SCAN_MS);
    }

    private synchronized void stopScan() {
        if (!scanning) return;
        scanning = false;
        handler.removeCallbacks(scanTimeout);
        BluetoothLeScanner scanner = adapter == null ? null : adapter.getBluetoothLeScanner();
        if (scanner != null) {
            try {
                scanner.stopScan(scanCallback);
            } catch (RuntimeException ignored) {
            }
        }
    }

    private final Runnable scanTimeout = new Runnable() {
        @Override public void run() {
            synchronized (BleEars.this) {
                stopScan();
                status = "not_found";
                if (!stopping) handler.postDelayed(BleEars.this::beginScan, 15_000L);
            }
        }
    };

    private final ScanCallback scanCallback = new ScanCallback() {
        @Override public void onScanResult(int callbackType, ScanResult result) {
            synchronized (BleEars.this) {
                if (!scanning || result == null) return;
                stopScan();
                address = result.getDevice().getAddress();
                connect(result.getDevice());
            }
        }

        @Override public void onScanFailed(int errorCode) {
            synchronized (BleEars.this) {
                scanning = false;
                status = "scan_failed_" + errorCode;
                if (!stopping) handler.postDelayed(BleEars.this::beginScan, 15_000L);
            }
        }
    };

    private long connectMs;
    private static final long STUCK_MS = 15_000L;       // connected but never streaming: start over

    private synchronized void connect(BluetoothDevice device) {
        connectMs = SystemClock.elapsedRealtime();
        status = "connecting";
        expected = -1;
        filled = 0;
        gatt = device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE);
    }

    private synchronized void restart(String why, long delayMs) {
        status = why;
        if (gatt != null) {
            // The link itself must go down before the next try (18.10): close() alone left Android's link up, the next
            // connectGatt reused it, and its second MTU request (allowed once per link) was never answered — every
            // command queued behind it and the ears stayed off. disconnect() first, close a second later.
            final BluetoothGatt old = gatt;
            gatt = null;
            try {
                old.disconnect();
            } catch (RuntimeException ignored) {
            }
            handler.postDelayed(() -> {
                try {
                    old.close();
                } catch (RuntimeException ignored) {
                }
            }, 1000L);
        }
        if (!stopping) handler.postDelayed(this::beginScan, Math.max(delayMs, 2500L));
    }

    private long ticks;
    private final Runnable watch = new Runnable() {
        @Override public void run() {
            synchronized (BleEars.this) {
                if (stopping) return;
                long now = SystemClock.elapsedRealtime();
                if (gatt != null && "streaming".equals(status) && now - lastDataMs > QUIET_MS) {
                    restart("silent", 500L);
                } else if (gatt != null && !"streaming".equals(status) && now - connectMs > STUCK_MS) {
                    // 18.10, the app reinstalled: «mtu» for good — the old process's link still open, no answer came
                    restart("stuck_" + status, 1000L);
                }
                if (++ticks % 10 == 0) Log.i("BleEars", statusJson());     // logcat: сверить с счётчиками платы
                handler.postDelayed(this, 1000L);
            }
        }
    };

    private final BluetoothGattCallback callback = new BluetoothGattCallback() {
        @Override public void onConnectionStateChange(BluetoothGatt g, int code, int state) {
            synchronized (BleEars.this) {
                if (g != gatt) return;
                if (code == BluetoothGatt.GATT_SUCCESS && state == BluetoothProfile.STATE_CONNECTED) {
                    status = "mtu";
                    g.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH);
                    if (!g.requestMtu(517)) {
                        status = "discovering";
                        g.discoverServices();
                    } else {
                        handler.postDelayed(() -> {
                            synchronized (BleEars.this) {
                                if (g == gatt && "mtu".equals(status)) {   // never answered: a whole new link
                                    restart("mtu_timeout", 2500L);
                                }
                            }
                        }, 3000L);
                    }
                } else {
                    restart(state == BluetoothProfile.STATE_DISCONNECTED ? "disconnected_" + code : "error_" + code,
                            2000L);
                }
            }
        }

        @Override public void onMtuChanged(BluetoothGatt g, int value, int code) {
            synchronized (BleEars.this) {
                if (g != gatt) return;
                mtu = value;
                status = "discovering";
                g.discoverServices();
            }
        }

        @Override public void onServicesDiscovered(BluetoothGatt g, int code) {
            synchronized (BleEars.this) {
                if (g != gatt) return;
                BluetoothGattService service = g.getService(SERVICE);
                BluetoothGattCharacteristic audio = service == null ? null : service.getCharacteristic(AUDIO);
                if (audio == null) { restart("no_service", 5000L); return; }
                g.setCharacteristicNotification(audio, true);
                BluetoothGattDescriptor cccd = audio.getDescriptor(CCCD);
                if (cccd == null) { restart("no_cccd", 5000L); return; }
                cccd.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                status = g.writeDescriptor(cccd) ? "subscribing" : "subscribe_failed";
            }
        }

        @Override public void onDescriptorWrite(BluetoothGatt g, BluetoothGattDescriptor d, int code) {
            synchronized (BleEars.this) {
                if (g != gatt) return;
                if (code == BluetoothGatt.GATT_SUCCESS) {
                    status = "streaming";
                    lastDataMs = SystemClock.elapsedRealtime();
                    BluetoothGattService service = g.getService(SERVICE);
                    BluetoothGattCharacteristic unlock = service == null ? null : service.getCharacteristic(UNLOCK);
                    if (unlock != null && !BuildConfig.EARS_TOKEN.isEmpty()) {
                        // after the subscription (one GATT operation at a time on Android)
                        unlock.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
                        unlock.setValue(BuildConfig.EARS_TOKEN.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                        if (!g.writeCharacteristic(unlock)) status = "unlock_failed";
                    }
                } else {
                    restart("subscribe_" + code, 3000L);      // encryption needed: Android pairs, then we try again
                }
            }
        }

        @Override public void onCharacteristicChanged(BluetoothGatt g, BluetoothGattCharacteristic c) {
            byte[] value = c.getValue();
            if (value != null) chunk(value);
        }
    };

    /** One notification: reassemble the board's packet and pass it on when whole. */
    private synchronized void chunk(byte[] value) {
        if (value.length < 1) return;
        chunks++;
        lastDataMs = SystemClock.elapsedRealtime();
        int head = value[0] & 0xff;
        int counter = head & 0x7f;
        boolean start = (head & 0x80) != 0;
        int offset = 1;
        if (start) {
            if (value.length < 3) return;
            if (expected >= 0) dropped++;                       // the previous packet never completed
            expected = (value[1] & 0xff) | ((value[2] & 0xff) << 8);
            filled = 0;
            offset = 3;
            if (expected <= 0 || expected > packet.length) { expected = -1; return; }
        } else if (expected < 0 || counter != ((lastCounter + 1) & 0x7f)) {
            if (expected >= 0) dropped++;
            expected = -1;                                      // a chunk was lost: wait for the next packet
            lastCounter = counter;
            return;
        }
        lastCounter = counter;
        int n = Math.min(value.length - offset, expected - filled);
        System.arraycopy(value, offset, packet, filled, n);
        filled += n;
        if (filled >= expected) {
            send(packet, expected);
            packets++;
            expected = -1;
        }
    }

    private void send(byte[] data, int length) {
        try {
            if (socket == null) {
                socket = new DatagramSocket();
                loopback = InetAddress.getByName("127.0.0.1");
            }
            socket.send(new DatagramPacket(data, length, loopback, LOCAL_PORT));
        } catch (Exception error) {
            status = "forward_failed";
        }
    }
}
