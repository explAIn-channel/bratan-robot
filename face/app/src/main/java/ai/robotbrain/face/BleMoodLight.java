package ai.robotbrain.face;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.AdvertiseCallback;
import android.bluetooth.le.AdvertiseData;
import android.bluetooth.le.AdvertiseSettings;
import android.bluetooth.le.BluetoothLeAdvertiser;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanRecord;
import android.bluetooth.le.ScanResult;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelUuid;
import android.os.SystemClock;
import android.util.SparseArray;

import java.security.SecureRandom;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

/** Direct local controller for the DLED Bluetooth mood light. */
@SuppressLint("MissingPermission")
final class BleMoodLight {
    private static final UUID SERVICE = UUID.fromString(
            "00010203-0405-0607-0809-0a0b0c0d1910");
    private static final UUID COMMAND = UUID.fromString(
            "00010203-0405-0607-0809-0a0b0c0d1912");
    private static final UUID PAIR = UUID.fromString(
            "00010203-0405-0607-0809-0a0b0c0d1914");
    private static final UUID ELK_SERVICE = UUID.fromString(
            "0000fff0-0000-1000-8000-00805f9b34fb");
    private static final UUID ELK_COMMAND = UUID.fromString(
            "0000fff3-0000-1000-8000-00805f9b34fb");
    private static final UUID ELK_SERVICE_ALT = UUID.fromString(
            "0000ffe5-0000-1000-8000-00805f9b34fb");
    private static final UUID ELK_COMMAND_ALT = UUID.fromString(
            "0000ffe9-0000-1000-8000-00805f9b34fb");
    private static final int TELINK_MANUFACTURER_ID = 529;
    private static final int ZENGGE_MANUFACTURER_ID = 63517;
    private static final int ELK_MANUFACTURER_ID = 48872;
    private static final long SCAN_WINDOW_MS = 12_000L;
    private static final long ELK_SETTLE_MS = 700L;
    // A write without response to the panel is on the air within a connection event; 0.9 s between them made
    // «listening» on the dock (power on, colour, brightness) take ~2 s (19.10, the owner: «подсветка далеко не сразу»).
    private static final long ELK_GAP_MS = 250L;
    private static final long REASSERT_MS = 60_000L;
    private static final long DOCK_PULSE_MS = 10_000L;
    private static final long DOCK_PULSE_TICK_MS = 250L;
    private static final long PREVIEW_MS = 10_000L;
    private static final int WRITE_POWER_ON = 1;
    private static final int WRITE_POWER_OFF = 2;
    private static final int WRITE_BRIGHTNESS = 3;
    private static final int WRITE_COLOR = 4;
    private static final int WRITE_EFFECT = 5;
    private static final int WRITE_SPEED = 6;
    private static final long EFFECT_MAX_MS = 15_000L;
    private static final MoodLighting.Profile DOCK_PROFILE =
            new MoodLighting.Profile("dock", 0, 180, 70, 0);

    private final Context context;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final SecureRandom random = new SecureRandom();
    private final BluetoothAdapter adapter;
    private BluetoothLeScanner scanner;
    private BluetoothLeAdvertiser advertiser;
    private BluetoothGatt gatt;
    private BluetoothGattCharacteristic pairCharacteristic;
    private BluetoothGattCharacteristic commandCharacteristic;
    private byte[] sessionRandom;
    private byte[] sessionKey;
    private final byte[] advertiseHost = new byte[2];
    private int advertiseSequence;
    private boolean elkProtocol;
    private int destination = MoodLighting.DEFAULT_MESH_UUID;
    private boolean scanning;
    private boolean stopping;
    private boolean writeInFlight;
    private long writeGeneration;
    private long writeDeadlineMs;
    private long acknowledgedGeneration;
    private String deliveryState = "unobserved";
    private boolean elkInitialized;
    private boolean previewNeedsPowerCycle;
    private boolean powerSent;
    private boolean offSent;
    private boolean docked;
    private boolean sleeping;
    private boolean dockPulseActive;
    private long dockPulseStartedMs;
    private int dockBrightness = 5;
    private int sentBrightness = -1;
    private int inFlightBrightness = -1;
    private int inFlightKind;
    private MoodLighting.Profile preview;
    private int previewBrightness = -1;
    private MoodLighting.Effect effect;     // a dynamic mode for a few seconds, then the feeling's own colour
    private boolean effectSent;
    private int sentSpeed = -1;
    private int inFlightSpeed = -1;
    private String inFlightState;
    private String sentState;
    private long sentAtMs;
    private MoodLighting.Profile desired = MoodLighting.forState("idle");
    private String status = "starting";
    private String address = "";
    private int rssi;
    private long seenCount;
    private String lastSeen = "";

    BleMoodLight(Context context) {
        this.context = context.getApplicationContext();
        BluetoothManager manager = (BluetoothManager)context.getSystemService(
                Context.BLUETOOTH_SERVICE);
        adapter = manager == null ? null : manager.getAdapter();
        random.nextBytes(advertiseHost);
    }

    synchronized void start() {
        stopping = false;
        beginScan();
    }

    synchronized void show(String emotion) {
        MoodLighting.Profile next = MoodLighting.forState(emotion);
        if (!next.state.equals(desired.state)) desired = next;
        if (sentAtMs != 0 && SystemClock.elapsedRealtime() - sentAtMs >= REASSERT_MS) {
            sentState = null;
        }
        flush();
    }

    /** The robot is asleep (its screen is dark): nothing inside may glow. */
    synchronized void setSleeping(boolean value) {
        if (sleeping == value) return;
        sleeping = value;
        offSent = false;
        sentState = null;
        sentBrightness = -1;
        if (!value) powerSent = false;          // waking: power on, then colour and brightness
        flush();
    }

    synchronized void setDocked(boolean value) {
        if (docked == value) return;
        docked = value;
        handler.removeCallbacks(dockPulse);
        dockPulseActive = value;
        offSent = false;
        sentState = null;
        sentBrightness = -1;
        if (value) {
            dockPulseStartedMs = SystemClock.elapsedRealtime();
            dockBrightness = 5;
            handler.post(dockPulse);
        } else {
            powerSent = false;
            flush();
        }
    }

    /** A feeling shimmers: the panel's own mode for `seconds`, even on the dock, then back to the colour. */
    synchronized boolean showEffect(String name, int seconds) {
        MoodLighting.Effect next = MoodLighting.effect(name);
        if (next == null || !isReady() || !elkProtocol || sleeping) return false;
        effect = next;
        effectSent = false;
        offSent = false;
        handler.removeCallbacks(endEffect);
        handler.postDelayed(endEffect, Math.max(1_000L, Math.min(EFFECT_MAX_MS, seconds * 1000L)));
        flush();
        return true;
    }

    private final Runnable endEffect = new Runnable() {
        @Override public void run() {
            synchronized (BleMoodLight.this) {
                effect = null;
                effectSent = false;
                sentState = null;            // a colour write ends the panel's mode
                sentBrightness = -1;
                offSent = false;             // resting on the dock: dark again
                flush();
            }
        }
    };

    synchronized boolean previewBrightness(int brightness) {
        if (!isReady() || brightness < 1 || brightness > 100) return false;
        preview = new MoodLighting.Profile("preview", 255, 255, 255, 0);
        previewBrightness = brightness;
        previewNeedsPowerCycle = true;
        offSent = false;
        sentBrightness = -1;
        handler.removeCallbacks(endPreview);
        handler.postDelayed(endPreview, PREVIEW_MS);
        flush();
        return true;
    }

    synchronized String healthJson() {
        String protocol = elkProtocol ? "elk_bledom" : "haodeng_telink";
        String expectedApp = elkProtocol ? "Lotus Lantern" : "Hao Deng";
        return "{\"protocol\":\"" + protocol + "\",\"expected_app\":\"" + expectedApp + "\"," +
                "\"status\":\"" + status + "\",\"ready\":" +
                isReady() + ",\"scanning\":" + scanning +
                ",\"device\":\"" + address + "\",\"rssi\":" + rssi +
                ",\"seen\":" + seenCount + ",\"last_seen\":\"" + lastSeen + "\"" +
                ",\"write_generation\":"+writeGeneration+",\"acknowledged_generation\":"+acknowledgedGeneration+
                ",\"delivery_state\":\""+deliveryState+"\",\"physical_color_verified\":false"+
                ",\"emotion\":\"" + desired.state + "\",\"brightness\":" +
                (preview != null ? previewBrightness :
                        MoodLighting.restingOnDock(docked, desired.state) ?
                                (dockPulseActive ? dockBrightness : 0) :
                                MoodLighting.brightnessForState(desired.state)) +
                ",\"docked\":" + docked + ",\"dock_pulse\":" + dockPulseActive +
                ",\"effect\":\"" + (effect == null ? "" : effect.name) + "\",\"effect_sent\":" + effectSent +
                ",\"speed\":" + sentSpeed + "}";
    }

    synchronized void shutdown() {
        stopping = true;
        handler.removeCallbacksAndMessages(null);
        stopScan();
        closeGatt();
        status = "stopped";
    }

    private synchronized void beginScan() {
        if (stopping || scanning || gatt != null) return;
        if (adapter == null) {
            status = "bluetooth_unavailable";
            return;
        }
        if (!hasPermission()) {
            status = "permission_missing";
            return;
        }
        if (!adapter.isEnabled()) {
            status = "enabling_bluetooth";
            adapter.enable();
            handler.postDelayed(this::beginScan, 1000L);
            return;
        }
        scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) {
            status = "scanner_unavailable";
            handler.postDelayed(this::beginScan, 2000L);
            return;
        }
        scanning = true;
        status = "scanning";
        scanner.startScan(scanCallback);
        handler.postDelayed(scanTimeout, SCAN_WINDOW_MS);
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= 31) {
            return context.checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) ==
                    PackageManager.PERMISSION_GRANTED &&
                    context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) ==
                            PackageManager.PERMISSION_GRANTED;
        }
        return context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED;
    }

    private final Runnable scanTimeout = new Runnable() {
        @Override public void run() {
            synchronized (BleMoodLight.this) {
                if (!scanning) return;
                stopScan();
                status = "not_found";
                if (!stopping) handler.postDelayed(BleMoodLight.this::beginScan, 2000L);
            }
        }
    };

    private final Runnable dockPulse = new Runnable() {
        @Override public void run() {
            synchronized (BleMoodLight.this) {
                if (!docked || !dockPulseActive) return;
                long elapsed = SystemClock.elapsedRealtime() - dockPulseStartedMs;
                if (elapsed >= DOCK_PULSE_MS) {
                    dockPulseActive = false;
                    flush();
                    return;
                }
                double phase = (elapsed % 5000L) / 5000.0;
                dockBrightness = 5 + (int)Math.round(25.0 *
                        (1.0 - Math.cos(phase * Math.PI * 2.0)) / 2.0);
                flush();
                handler.postDelayed(this, DOCK_PULSE_TICK_MS);
            }
        }
    };

    private final Runnable endPreview = new Runnable() {
        @Override public void run() {
            synchronized (BleMoodLight.this) {
                preview = null;
                previewBrightness = -1;
                sentState = null;
                sentBrightness = -1;
                flush();
            }
        }
    };

    private final ScanCallback scanCallback = new ScanCallback() {
        @Override public void onScanResult(int callbackType, ScanResult result) {
            accept(result);
        }

        @Override public void onBatchScanResults(List<ScanResult> results) {
            for (ScanResult result : results) accept(result);
        }

        @Override public void onScanFailed(int errorCode) {
            synchronized (BleMoodLight.this) {
                scanning = false;
                status = "scan_failed_" + errorCode;
                if (!stopping) handler.postDelayed(BleMoodLight.this::beginScan, 2000L);
            }
        }
    };

    private synchronized void accept(ScanResult result) {
        if (stopping || !scanning || result == null) return;
        ScanRecord record = result.getScanRecord();
        seenCount++;
        SparseArray<byte[]> values = record == null ? null : record.getManufacturerSpecificData();
        StringBuilder ids = new StringBuilder();
        if (values != null) {
            for (int i = 0; i < values.size(); i++) {
                if (i > 0) ids.append(',');
                ids.append(values.keyAt(i));
            }
        }
        String name = result.getDevice().getName();
        lastSeen = ((name == null ? "" : name) + "@" + result.getDevice().getAddress() +
                "#" + ids).replace("\\", "\\\\").replace("\"", "\\\"");
        byte[] manufacturer = record == null ? null :
                record.getManufacturerSpecificData(TELINK_MANUFACTURER_ID);
        byte[] zengge = record == null ? null :
                record.getManufacturerSpecificData(ZENGGE_MANUFACTURER_ID);
        boolean serviceSeen = false;
        if (record != null && record.getServiceUuids() != null) {
            for (ParcelUuid uuid : record.getServiceUuids()) {
                if (SERVICE.equals(uuid.getUuid())) serviceSeen = true;
            }
        }
        boolean elkName = name != null && name.toUpperCase(Locale.US).startsWith("ELK-BLED");
        if ((manufacturer == null || manufacturer.length < 10) &&
                zengge == null && !serviceSeen && !elkName) return;

        if (manufacturer != null && manufacturer.length >= 2) {
            int advertisedMesh = (manufacturer[0] & 0xff) | ((manufacturer[1] & 0xff) << 8);
            if (advertisedMesh != 0) destination = advertisedMesh;
        }
        if (manufacturer != null && manufacturer.length >= 10 &&
                (manufacturer[9] & 0xff) != 0) {
            destination = manufacturer[9] & 0xff;
        }
        address = result.getDevice().getAddress().toUpperCase(Locale.US);
        rssi = result.getRssi();
        elkProtocol = elkName;
        stopScan();
        status = "connecting";
        gatt = result.getDevice().connectGatt(context, false, gattCallback,
                BluetoothDevice.TRANSPORT_LE);
    }

    private final BluetoothGattCallback gattCallback = new BluetoothGattCallback() {
        @Override public void onConnectionStateChange(BluetoothGatt callbackGatt,
                                                       int callbackStatus, int newState) {
            synchronized (BleMoodLight.this) {
                if (callbackGatt != gatt) return;
                if (callbackStatus == BluetoothGatt.GATT_SUCCESS &&
                        newState == android.bluetooth.BluetoothProfile.STATE_CONNECTED) {
                    elkInitialized = !elkProtocol;
                    status = elkProtocol ? "initializing" : "discovering";
                    if (!gatt.discoverServices()) retry("service_discovery_failed", 1500L);
                    handler.postDelayed(() -> {
                        synchronized (BleMoodLight.this) {
                            if (gatt == callbackGatt && !stopping) {
                                callbackGatt.requestMtu(63);
                                callbackGatt.discoverServices();
                                handler.postDelayed(() -> {
                                    synchronized (BleMoodLight.this) {
                                        if (gatt == callbackGatt && !stopping) {
                                            elkInitialized = true;
                                            if (commandCharacteristic != null) {
                                                status = "ready";
                                                flush();
                                            }
                                        }
                                    }
                                }, ELK_SETTLE_MS);
                            }
                        }
                    }, 1000L);
                    handler.postDelayed(() -> {
                        synchronized (BleMoodLight.this) {
                            if (gatt == callbackGatt && !isReady() && !stopping) {
                                retry("initialization_timeout", 1200L);
                            }
                        }
                    }, 6000L);
                } else if (newState == android.bluetooth.BluetoothProfile.STATE_DISCONNECTED) {
                    retry("disconnected_" + callbackStatus, 1200L);
                }
            }
        }

        @Override public void onServicesDiscovered(BluetoothGatt callbackGatt,
                                                    int callbackStatus) {
            synchronized (BleMoodLight.this) {
                if (callbackGatt != gatt) return;
                if (callbackStatus != BluetoothGatt.GATT_SUCCESS) {
                    retry("service_discovery_error_" + callbackStatus, 1500L);
                    return;
                }
                BluetoothGattService service = gatt.getService(SERVICE);
                if (elkProtocol) {
                    BluetoothGattService elkService = gatt.getService(ELK_SERVICE);
                    UUID commandUuid = ELK_COMMAND;
                    if (elkService == null) {
                        elkService = gatt.getService(ELK_SERVICE_ALT);
                        commandUuid = ELK_COMMAND_ALT;
                    }
                    if (elkService == null ||
                            (commandCharacteristic = elkService.getCharacteristic(commandUuid)) == null) {
                        retry("elk_service_missing", 3000L);
                        return;
                    }
                    powerSent = false;
                    sentState = null;
                    if (!elkInitialized) {
                        status = "initializing";
                        return;
                    }
                    status = "ready";
                    flush();
                    return;
                }
                if (service == null || (pairCharacteristic = service.getCharacteristic(PAIR)) == null ||
                        (commandCharacteristic = service.getCharacteristic(COMMAND)) == null) {
                    retry("haodeng_service_missing", 3000L);
                    return;
                }
                sessionRandom = new byte[8];
                random.nextBytes(sessionRandom);
                pairCharacteristic.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
                pairCharacteristic.setValue(MoodLighting.pairPacket(sessionRandom));
                status = "authenticating";
                if (!gatt.writeCharacteristic(pairCharacteristic)) {
                    retry("pair_write_rejected", 1500L);
                }
            }
        }

        @Override public void onCharacteristicWrite(BluetoothGatt callbackGatt,
                                                     BluetoothGattCharacteristic characteristic,
                                                     int callbackStatus) {
            synchronized (BleMoodLight.this) {
                if (callbackGatt != gatt) return;
                if (callbackStatus != BluetoothGatt.GATT_SUCCESS) {
                    retry("write_error_" + callbackStatus, 1500L);
                    return;
                }
                if (PAIR.equals(characteristic.getUuid())) {
                    handler.postDelayed(() -> {
                        synchronized (BleMoodLight.this) {
                            if (gatt == callbackGatt && !stopping &&
                                    !gatt.readCharacteristic(pairCharacteristic)) {
                                retry("pair_read_rejected", 1500L);
                            }
                        }
                    }, 300L);
                    return;
                }
                if (elkProtocol) return;
                if (!writeInFlight || !COMMAND.equals(characteristic.getUuid()) ||
                        SystemClock.elapsedRealtime()>writeDeadlineMs) return;
                acknowledgedGeneration=writeGeneration;
                deliveryState="transport_acknowledged_physical_unverified";
                finishWrite();
            }
        }

        @Override public void onCharacteristicRead(BluetoothGatt callbackGatt,
                                                    BluetoothGattCharacteristic characteristic,
                                                    int callbackStatus) {
            synchronized (BleMoodLight.this) {
                if (callbackGatt != gatt) return;
                if (!PAIR.equals(characteristic.getUuid()) ||
                        callbackStatus != BluetoothGatt.GATT_SUCCESS) {
                    retry("pair_read_error_" + callbackStatus, 2000L);
                    return;
                }
                byte[] reply = characteristic.getValue();
                if (reply == null || reply.length < 9 || reply[0] != 0x0d) {
                    retry("authentication_failed", 10_000L);
                    return;
                }
                byte[] responseRandom = new byte[8];
                System.arraycopy(reply, 1, responseRandom, 0, 8);
                sessionKey = MoodLighting.sessionKey(sessionRandom, responseRandom);
                powerSent = false;
                sentState = null;
                status = "ready";
                flush();
            }
        }
    };

    private synchronized void flush() {
        if (!isReady() || gatt == null || commandCharacteristic == null ||
                writeInFlight) return;
        if (preview != null) {
            if (previewNeedsPowerCycle) {
                previewNeedsPowerCycle = false;
                write(MoodLighting.elkPowerPacket(false), null, WRITE_POWER_OFF, -1);
            } else if (!powerSent) {
                write(MoodLighting.elkPowerPacket(true), null, WRITE_POWER_ON, -1);
            } else if (!preview.state.equals(sentState)) {
                write(MoodLighting.elkColorPacket(preview), preview.state,
                        WRITE_COLOR, -1);
            } else if (sentBrightness != previewBrightness) {
                write(MoodLighting.elkBrightnessPacket(previewBrightness), null,
                        WRITE_BRIGHTNESS, previewBrightness);
            }
            return;
        }
        if (sleeping) {
            if (!offSent) write(elkProtocol ? MoodLighting.elkPowerPacket(false) :
                    MoodLighting.powerPacket(sessionKey, address, destination, false, sequence()),
                    null, WRITE_POWER_OFF, -1);
            return;
        }
        if (effect != null && elkProtocol) {
            // The mode first (seen at once), its speed next; a later colour write ends it.
            if (!powerSent) {
                write(MoodLighting.elkPowerPacket(true), null, WRITE_POWER_ON, -1);
            } else if (!effectSent) {
                write(MoodLighting.elkEffectPacket(effect.code), "effect:" + effect.name, WRITE_EFFECT, -1);
            } else if (sentSpeed != effect.speed) {
                inFlightSpeed = effect.speed;
                write(MoodLighting.elkSpeedPacket(effect.speed), null, WRITE_SPEED, -1);
            }
            return;
        }
        if (MoodLighting.restingOnDock(docked, desired.state) && elkProtocol) {
            if (!dockPulseActive) {
                if (!offSent) write(MoodLighting.elkPowerPacket(false), null,
                        WRITE_POWER_OFF, -1);
                return;
            }
            if (!powerSent) {
                write(MoodLighting.elkPowerPacket(true), null, WRITE_POWER_ON, -1);
            } else if (!DOCK_PROFILE.state.equals(sentState)) {
                write(MoodLighting.elkColorPacket(DOCK_PROFILE), DOCK_PROFILE.state,
                        WRITE_COLOR, -1);
            } else if (sentBrightness != dockBrightness) {
                write(MoodLighting.elkBrightnessPacket(dockBrightness), null,
                        WRITE_BRIGHTNESS, dockBrightness);
            }
            return;
        }
        if (!powerSent) {
            write(elkProtocol ? MoodLighting.elkPowerPacket(true) :
                    MoodLighting.powerPacket(sessionKey, address, destination, true,
                            sequence()), null, WRITE_POWER_ON, -1);
        } else if (desired != null && !desired.state.equals(sentState)) {
            write(elkProtocol ? MoodLighting.elkColorPacket(desired) :
                    MoodLighting.colorPacket(sessionKey, address, destination, desired,
                            sequence()), desired.state, WRITE_COLOR, -1);
        } else if (elkProtocol && sentBrightness !=
                MoodLighting.brightnessForState(desired.state)) {
            int brightness = MoodLighting.brightnessForState(desired.state);
            write(MoodLighting.elkBrightnessPacket(brightness), null,
                    WRITE_BRIGHTNESS, brightness);
        }
    }

    private void write(byte[] value, String state, int kind, int brightness) {
        if (elkProtocol) advertise(value);
        commandCharacteristic.setWriteType(elkProtocol ?
                BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE :
                BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
        commandCharacteristic.setValue(value);
        inFlightState = state;
        inFlightKind = kind;
        inFlightBrightness = brightness;
        writeInFlight = true;
        final long ownedWrite=++writeGeneration;
        final BluetoothGatt writeGatt=gatt;
        writeDeadlineMs=SystemClock.elapsedRealtime()+2500L;
        deliveryState="submitted_physical_unverified";
        handler.postDelayed(() -> {
            synchronized (BleMoodLight.this) {
                if (gatt==writeGatt && writeInFlight && writeGeneration==ownedWrite) {
                    deliveryState="transport_timeout";
                    retry("command_write_timeout",1500L); // Close this GATT; late callbacks cannot finish a new write.
                }
            }
        },2500L);
        if (!gatt.writeCharacteristic(commandCharacteristic)) {
            writeInFlight = false;
            inFlightState = null;
            inFlightKind = 0;
            inFlightBrightness = -1;
            retry("command_write_rejected", 1500L);
        } else if (elkProtocol) {
            handler.postDelayed(() -> {
                synchronized (BleMoodLight.this) {
                    if (gatt == writeGatt && writeInFlight && writeGeneration==ownedWrite) {
                        deliveryState="submitted_without_response_physical_unverified";
                        finishWrite();
                    }
                }
            }, ELK_GAP_MS);
        }
    }

    private void advertise(byte[] value) {
        advertiser = adapter == null ? null : adapter.getBluetoothLeAdvertiser();
        if (advertiser == null) return;
        byte[] packet = MoodLighting.elkAdvertisePacket(value,
                advertiseSequence++ & 0xff, advertiseHost[0], advertiseHost[1]);
        advertiser.stopAdvertising(advertiseCallback);
        advertiser.startAdvertising(new AdvertiseSettings.Builder()
                        .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
                        .setConnectable(false)
                        .setTimeout(0)
                        .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
                        .build(),
                new AdvertiseData.Builder()
                        .addManufacturerData(ELK_MANUFACTURER_ID, packet)
                        .setIncludeDeviceName(false)
                        .setIncludeTxPowerLevel(false)
                        .build(), advertiseCallback);
    }

    private final AdvertiseCallback advertiseCallback = new AdvertiseCallback() {};

    private synchronized void finishWrite() {
        if (!writeInFlight) return;
        writeInFlight = false;
        switch (inFlightKind) {
            case WRITE_POWER_ON:
                powerSent = true;
                offSent = false;
                break;
            case WRITE_POWER_OFF:
                powerSent = false;
                offSent = true;
                break;
            case WRITE_BRIGHTNESS:
                sentBrightness = inFlightBrightness;
                break;
            case WRITE_COLOR:
                sentState = inFlightState;
                sentAtMs = SystemClock.elapsedRealtime();
                break;
            case WRITE_EFFECT:
                effectSent = effect != null;
                sentState = inFlightState;       // not a feeling's colour: it is written again afterwards
                break;
            case WRITE_SPEED:
                sentSpeed = inFlightSpeed;
                break;
            default:
                break;
        }
        inFlightSpeed = -1;
        inFlightState = null;
        inFlightKind = 0;
        inFlightBrightness = -1;
        flush();
    }

    private boolean isReady() {
        return elkProtocol ? commandCharacteristic != null && "ready".equals(status) :
                sessionKey != null;
    }

    private byte[] sequence() {
        byte[] value = new byte[3];
        random.nextBytes(value);
        return value;
    }

    private synchronized void retry(String reason, long delayMs) {
        status = reason;
        closeGatt();
        if (!stopping) handler.postDelayed(this::beginScan, delayMs);
    }

    private void stopScan() {
        handler.removeCallbacks(scanTimeout);
        if (scanning && scanner != null) scanner.stopScan(scanCallback);
        scanning = false;
    }

    private void closeGatt() {
        ++writeGeneration;
        sessionKey = null;
        elkProtocol = false;
        elkInitialized = false;
        previewNeedsPowerCycle = false;
        pairCharacteristic = null;
        commandCharacteristic = null;
        writeInFlight = false;
        powerSent = false;
        offSent = false;
        sentBrightness = -1;
        inFlightBrightness = -1;
        inFlightKind = 0;
        inFlightState = null;
        sentState = null;
        effectSent = false;
        sentSpeed = -1;
        inFlightSpeed = -1;
        if (advertiser != null) advertiser.stopAdvertising(advertiseCallback);
        advertiser = null;
        if (gatt != null) {
            gatt.disconnect();
            gatt.close();
            gatt = null;
        }
    }
}
