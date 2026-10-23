package ai.robotbrain.face;

import android.annotation.SuppressLint;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.audiofx.Visualizer;
import android.media.audiofx.LoudnessEnhancer;
import android.media.MediaDataSource;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
import android.os.SystemClock;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;
import java.util.LinkedHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import android.os.Handler;
import android.os.Looper;

/** Local-only microphone/speaker bridge for STT/TTS. No network or motion API. */
public final class AudioIO {
    private static final int SAMPLE_RATE = 16000;
    private static final int WINDOW_SECONDS = 12;
    static final int SPEECH_BOOST_MB = 1050;   // +10.5 dB: owner asked for max, then «на 15% тише» (06.10.2026)
    private static final int MAX_STREAM_BYTES = SAMPLE_RATE * 2 * 3;
    private final Object lock = new Object();
    private final byte[] ring = new byte[SAMPLE_RATE * 2 * WINDOW_SECONDS];
    private int writeOffset;
    private volatile long totalBytes;
    private final String producerInstance = UUID.randomUUID().toString();
    private volatile long captureGeneration;
    private volatile long captureStartedMs;
    private volatile long nextCaptureAttemptMs;
    private volatile int captureRetries;
    private volatile boolean running;
    private volatile String lastError = "not started";
    private volatile double rms;
    private volatile long lastAudioMs;
    private volatile long lastPlaybackMs;
    private AudioRecord record;
    private Thread recordThread;
    private NoiseSuppressor noiseSuppressor;
    private AutomaticGainControl automaticGainControl;
    private AcousticEchoCanceler acousticEchoCanceler;
    private volatile boolean noiseSuppressionEnabled;
    private volatile boolean automaticGainEnabled;
    private volatile boolean echoCancellationEnabled;
    // 08.10: Android's noise suppressor woke up late after silence and ate the first word of a phrase -
    // the robot's name («Братан, съезди к кровати» reached the ears as «съезди к кровати»). Off by default,
    // switchable at run time (POST /audio/effects?ns=0|1&aec=0|1) for comparisons.
    // On by default (face 89): off, the room's music and talk kept the name's gate open and «Братан» was lost
    // (08.10 20:50); the voice process re-asserts it every 5 min, but a restarted app must not start deaf.
    private volatile boolean wantNoiseSuppression = true;
    private volatile boolean wantEchoCancellation = true;
    private final Object playbackLock = new Object();
    final LiveAudio liveAudio = new LiveAudio(this::speech);
    String startLive(String id) {
        synchronized(playbackLock) {
            if(playbackThread!=null && playbackThread.isAlive()) return null;
            return liveAudio.start(id);
        }
    }
    private MediaPlayer activeCompressedPlayer;
    private long playbackGeneration;
    private final FaceView faceView;
    private volatile boolean playing;
    private volatile boolean lipSyncReady;
    private volatile float speechLevel;
    private volatile String playbackError = "";
    private final LinkedHashMap<String, Playback> playbacks = new LinkedHashMap<>();
    private Thread playbackThread;
    private final Handler playbackWatchdog = new Handler(Looper.getMainLooper());

    AudioIO(FaceView faceView) { this.faceView = faceView; }

    void meter(float level, boolean active) { speech(level, active); }

    private void speech(float level, boolean active) {
        speechLevel = level;
        playing = active;
        faceView.setSpeechLevel(level, active);
    }

    @SuppressLint("MissingPermission")
    synchronized void start() {
        if (running || (recordThread != null && recordThread.isAlive()) ||
                SystemClock.elapsedRealtime() < nextCaptureAttemptMs) return;
        final long generation = ++captureGeneration;
        synchronized (lock) { totalBytes = 0; writeOffset = 0; lastAudioMs = 0; }
        captureStartedMs = SystemClock.elapsedRealtime();
        int minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (minimum <= 0) minimum = SAMPLE_RATE;
        try {
            record = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, Math.max(minimum * 2, SAMPLE_RATE));
            if (record.getState() != AudioRecord.STATE_INITIALIZED) {
                throw new IllegalStateException("AudioRecord initialization failed");
            }
            enableCaptureEffects(record.getAudioSessionId());
            running = true;
            lastError = "";
            record.startRecording();
            final AudioRecord ownedRecord = record;
            recordThread = new Thread(() -> recordLoop(ownedRecord, generation), "robotbrain-microphone");
            recordThread.setDaemon(true);
            recordThread.start();
        } catch (Exception error) {
            lastError = error.getClass().getSimpleName() + ": " + safe(error.getMessage());
            running = false;
            if (record != null) { try { record.release(); } catch (RuntimeException ignored) { } }
            record = null;
            releaseCaptureEffects();
            captureRetries++;
            nextCaptureAttemptMs = SystemClock.elapsedRealtime()+Math.min(30000L,1000L << Math.min(captureRetries,5));
        }
    }

    /** Enable Android's hardware/vendor speech front-end when available. */
    private void enableCaptureEffects(int sessionId) {
        releaseCaptureEffects();
        try {
            if (NoiseSuppressor.isAvailable()) {
                noiseSuppressor = NoiseSuppressor.create(sessionId);
                if (noiseSuppressor != null) {
                    noiseSuppressor.setEnabled(wantNoiseSuppression);
                    noiseSuppressionEnabled = noiseSuppressor.getEnabled();
                }
            }
        } catch (RuntimeException ignored) { noiseSuppressionEnabled = false; }
        try {
            if (AutomaticGainControl.isAvailable()) {
                automaticGainControl = AutomaticGainControl.create(sessionId);
                if (automaticGainControl != null) {
                    automaticGainControl.setEnabled(true);
                    automaticGainEnabled = automaticGainControl.getEnabled();
                }
            }
        } catch (RuntimeException ignored) { automaticGainEnabled = false; }
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                acousticEchoCanceler = AcousticEchoCanceler.create(sessionId);
                if (acousticEchoCanceler != null) {
                    acousticEchoCanceler.setEnabled(wantEchoCancellation);
                    echoCancellationEnabled = acousticEchoCanceler.getEnabled();
                }
            }
        } catch (RuntimeException ignored) { echoCancellationEnabled = false; }
    }

    /** Switch the capture effects of the running microphone (null = leave as is). */
    synchronized String setEffects(Boolean noise, Boolean echo) {
        if (noise != null) wantNoiseSuppression = noise;
        if (echo != null) wantEchoCancellation = echo;
        try {
            if (noiseSuppressor != null) {
                noiseSuppressor.setEnabled(wantNoiseSuppression);
                noiseSuppressionEnabled = noiseSuppressor.getEnabled();
            }
        } catch (RuntimeException ignored) { }
        try {
            if (acousticEchoCanceler != null) {
                acousticEchoCanceler.setEnabled(wantEchoCancellation);
                echoCancellationEnabled = acousticEchoCanceler.getEnabled();
            }
        } catch (RuntimeException ignored) { }
        return "{\"ok\":true,\"noise_suppression_enabled\":" + noiseSuppressionEnabled +
                ",\"echo_cancellation_enabled\":" + echoCancellationEnabled + "}";
    }

    private void releaseCaptureEffects() {
        if (noiseSuppressor != null) try { noiseSuppressor.release(); } catch (RuntimeException ignored) { }
        if (automaticGainControl != null) try { automaticGainControl.release(); } catch (RuntimeException ignored) { }
        if (acousticEchoCanceler != null) try { acousticEchoCanceler.release(); } catch (RuntimeException ignored) { }
        noiseSuppressor = null;
        automaticGainControl = null;
        acousticEchoCanceler = null;
        noiseSuppressionEnabled = false;
        automaticGainEnabled = false;
        echoCancellationEnabled = false;
    }

    private void recordLoop(AudioRecord ownedRecord, long generation) {
        // 20 ms reads: microphone frames reach the robot's ears without a 128 ms batch delay.
        byte[] buffer = new byte[640];
        int emptyReads = 0;
        try { while (running && generation == captureGeneration) {
            int count = ownedRecord.read(buffer, 0, buffer.length);
            if (count <= 0) {
                lastError = "AudioRecord read code " + count;
                if (count < 0 || ++emptyReads >= 20) break;
                SystemClock.sleep(50);
                continue;
            }
            emptyReads = 0;
            count -= count & 1;
            double sum = 0;
            int samples = 0;
            for (int i = 0; i + 1 < count; i += 2) {
                short value = (short)((buffer[i] & 0xff) | (buffer[i + 1] << 8));
                double normalized = value / 32768.0;
                sum += normalized * normalized;
                samples++;
            }
            rms = samples == 0 ? 0 : Math.sqrt(sum / samples);
            synchronized (lock) {
                if (!running || generation != captureGeneration) break;
                int source = 0;
                while (source < count) {
                    int chunk = Math.min(count - source, ring.length - writeOffset);
                    System.arraycopy(buffer, source, ring, writeOffset, chunk);
                    source += chunk;
                    writeOffset = (writeOffset + chunk) % ring.length;
                    totalBytes += chunk;
                }
            }
            lastAudioMs = SystemClock.elapsedRealtime();
            lastError = "";
            captureRetries = 0;
        }} catch (RuntimeException error) {
            if (generation == captureGeneration) lastError = "capture_failed:"+error.getClass().getSimpleName();
        } finally {
            try { ownedRecord.stop(); } catch (RuntimeException ignored) { }
            try { ownedRecord.release(); } catch (RuntimeException ignored) { }
            synchronized (this) {
                if (record == ownedRecord) { record = null; releaseCaptureEffects(); }
                if (recordThread == Thread.currentThread()) recordThread = null;
                if (generation == captureGeneration) {
                    running = false;
                    captureRetries++;
                    nextCaptureAttemptMs=SystemClock.elapsedRealtime()+Math.min(30000L,1000L << Math.min(captureRetries,5));
                }
            }
        }
    }

    void recoverCapture(boolean permissionGranted) {
        if (!permissionGranted) { stopCapture(); lastError="microphone_permission_denied"; return; }
        long last = lastAudioMs > 0 ? lastAudioMs : captureStartedMs;
        if (running && SystemClock.elapsedRealtime()-last > 2000) stopCapture();
        if (!running) start();
    }

    String producerInstance() { return producerInstance; }
    long captureGeneration() { return captureGeneration; }
    long lastAudioMs() { return lastAudioMs; }

    private void stopCapture() {
        AudioRecord ownedRecord;
        Thread ownedThread;
        synchronized (this) {
            running=false;
            ++captureGeneration;
            ownedRecord=record;
            ownedThread=recordThread;
        }
        if (ownedRecord != null) try { ownedRecord.stop(); } catch (RuntimeException ignored) { }
        if (ownedThread != null && ownedThread != Thread.currentThread()) {
            try { ownedThread.join(500); } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        }
        synchronized (this) {
            if (ownedThread != null && ownedThread.isAlive()) {
                lastError="capture_stop_pending";
                return; // Never open a second owner over a wedged capture thread.
            }
            if (record == ownedRecord && record != null) {
                try { record.release(); } catch (RuntimeException ignored) { }
                record=null;
                releaseCaptureEffects();
            }
            if (recordThread == ownedThread) recordThread=null;
        }
    }

    boolean isReady() {
        return running && totalBytes >= SAMPLE_RATE * 2 &&
                SystemClock.elapsedRealtime() - lastAudioMs < 1000;
    }

    synchronized boolean isCaptureActive() {
        try { return running && record != null &&
                record.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING; }
        catch (RuntimeException ignored) { return false; }
    }

    byte[] latestWav() {
        if (!isReady()) return null;
        byte[] pcm;
        synchronized (lock) {
            int length = (int)Math.min(totalBytes, ring.length);
            if (length < SAMPLE_RATE * 2) return null;
            pcm = new byte[length];
            int start = (writeOffset - length + ring.length) % ring.length;
            int first = Math.min(length, ring.length - start);
            System.arraycopy(ring, start, pcm, 0, first);
            if (first < length) System.arraycopy(ring, 0, pcm, first, length - first);
        }
        return wav(pcm, SAMPLE_RATE, 1, 16);
    }

    long cursor() {
        synchronized (lock) { return totalBytes; }
    }

    static final class PcmChunk {
        final byte[] pcm;
        final long startCursor;
        final long endCursor;
        final boolean dropped;
        final long generation;
        final long frameElapsedMs;

        PcmChunk(byte[] pcm, long startCursor, long endCursor, boolean dropped, long generation, long frameElapsedMs) {
            this.pcm = pcm;
            this.startCursor = startCursor;
            this.endCursor = endCursor;
            this.dropped = dropped;
            this.generation = generation;
            this.frameElapsedMs = frameElapsedMs;
        }
    }

    /** Return each captured sample at most once to a localhost consumer. */
    PcmChunk readSince(long requestedCursor, int requestedMaxBytes) {
        synchronized (lock) {
            long end = totalBytes;
            long oldest = Math.max(0, end - ring.length);
            long requested = Math.max(0, requestedCursor);
            long start = Math.max(oldest, Math.min(requested, end));
            int maximum = Math.max(2, Math.min(requestedMaxBytes, MAX_STREAM_BYTES));
            maximum -= maximum & 1;
            int length = (int)Math.min(end - start, maximum);
            length -= length & 1;
            byte[] pcm = new byte[length];
            int ringStart = (int)(start % ring.length);
            int first = Math.min(length, ring.length - ringStart);
            System.arraycopy(ring, ringStart, pcm, 0, first);
            if (first < length) System.arraycopy(ring, 0, pcm, first, length - first);
            long frameElapsedMs=lastAudioMs-(end-start-length)*1000L/(SAMPLE_RATE*2);
            return new PcmChunk(pcm, start, start + length, requested < oldest, captureGeneration, frameElapsedMs);
        }
    }


    private static final class Playback {
        final String id;
        final long generation;
        final long acceptedMs = SystemClock.elapsedRealtime();
        String state = "accepted";
        long startedMs;
        long terminalMs;
        String error = "";
        byte[] fingerprint;
        Playback(String id, long generation) { this.id=id; this.generation=generation; }
        boolean terminal() { return state.equals("completed") || state.equals("failed") || state.equals("cancelled"); }
    }

    String acceptPlayback(byte[] source, String id, boolean mp3) {
        if (id == null || !id.matches("[A-Za-z0-9._:-]{8,128}")) return null;
        if (source == null || source.length > 8*1024*1024) return null;
        if (mp3) {
            if (source.length < 128) return null;
            boolean id3=source[0]=='I' && source[1]=='D' && source[2]=='3';
            boolean frame=(source[0]&0xff)==0xff && (source[1]&0xe0)==0xe0;
            if (!id3 && !frame) return null;
        } else {
            if (source.length <= 44) return null;
            ByteBuffer header=ByteBuffer.wrap(source).order(ByteOrder.LITTLE_ENDIAN);
            if (header.getInt(0)!=0x46464952 || header.getInt(8)!=0x45564157) return null;
        }
        final Playback item;
        final byte[] encoded;
        final byte[] fingerprint;
        try { fingerprint=MessageDigest.getInstance("SHA-256").digest(source); }
        catch (NoSuchAlgorithmException unavailable) { return null; }
        synchronized (playbackLock) {
            if (playbacks.containsKey(id)) {
                Playback existing=playbacks.get(id);
                if (existing.fingerprint != null && !Arrays.equals(existing.fingerprint,fingerprint)) return null;
                return playbackJsonLocked(existing);
            }
            if (liveAudio.active() || (playbackThread != null && playbackThread.isAlive())) return null;
            prunePlaybackLocked();
            if (playbacks.size() >= 128) return null;
            item=new Playback(id,++playbackGeneration);
            item.fingerprint=fingerprint;
            playbacks.put(id,item);
            encoded=Arrays.copyOf(source,source.length);
            playbackThread=new Thread(() -> playbackLoop(encoded,item),"robotbrain-speaker");
            playbackThread.setDaemon(true);
            playbackThread.start();
            return playbackJsonLocked(item);
        }
    }

    private void prunePlaybackLocked() {
        long cutoff=SystemClock.elapsedRealtime()-120_000L;
        java.util.Iterator<Playback> iterator=playbacks.values().iterator();
        while (iterator.hasNext()) {
            Playback item=iterator.next();
            if (item.terminal() && item.terminalMs < cutoff) iterator.remove();
        }
    }

    String playbackJson(String id) {
        synchronized (playbackLock) {
            Playback item=playbacks.get(id);
            return item == null ? null : playbackJsonLocked(item);
        }
    }

    private String playbackJsonLocked(Playback item) {
        return "{\"ok\":true,\"accepted\":true,\"playback_id\":\""+json(item.id)+
            "\",\"producer_instance\":\""+producerInstance+"\",\"generation\":"+item.generation+
            ",\"state\":\""+item.state+"\",\"accepted_elapsed_ms\":"+item.acceptedMs+
            ",\"started_elapsed_ms\":"+item.startedMs+",\"terminal_elapsed_ms\":"+item.terminalMs+
            ",\"error\":\""+json(item.error)+"\"}";
    }

    String cancelPlayback(String id) {
        if (id == null || !id.matches("[A-Za-z0-9._:-]{8,128}")) return null;
        synchronized (playbackLock) {
            Playback item=playbacks.get(id);
            if (item == null) {
                prunePlaybackLocked();
                if (playbacks.size() >= 128) return null;
                item=new Playback(id,0);
                playbacks.put(id,item); // Tombstone a cancellation racing upload.
            }
            if (!item.terminal()) {
                item.state="cancelled";
                item.terminalMs=SystemClock.elapsedRealtime();
                if (item.generation == playbackGeneration && item.generation > 0) {
                    ++playbackGeneration;
                    if (activeCompressedPlayer != null) {
                        try { activeCompressedPlayer.reset(); } catch (RuntimeException ignored) { }
                    }
                    speech(0f,false);
                }
            }
            return playbackJsonLocked(item);
        }
    }

    private void playbackLoop(byte[] encoded, Playback item) {
        MediaPlayer player=null;
        Visualizer visualizer=null;
        LoudnessEnhancer louder=null;
        AtomicBoolean completed=new AtomicBoolean();
        AtomicBoolean failed=new AtomicBoolean();
        Runnable expire=() -> {
            synchronized (playbackLock) {
                if (!item.terminal()) {
                    cancelPlayback(item.id);
                    item.state="failed";
                    item.error="playback_deadline";
                }
            }
        };
        playbackWatchdog.postDelayed(expire,120_000L);
        try {
            player=new MediaPlayer();
            synchronized (playbackLock) {
                if (item.terminal() || item.generation!=playbackGeneration) return;
                activeCompressedPlayer=player;
            }
            player.setOnCompletionListener(ignored -> completed.set(true));
            player.setOnErrorListener((ignored,what,extra) -> { failed.set(true); return true; });
            player.setAudioAttributes(new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build());
            player.setDataSource(new ByteArrayMediaDataSource(encoded));
            player.prepare();
            // The OpenAI voices are mastered quieter than the old Microsoft one and the
            // speaker volume is already at maximum: boost with the built-in limiter.
            try {
                louder=new LoudnessEnhancer(player.getAudioSessionId());
                louder.setTargetGain(SPEECH_BOOST_MB);
                louder.setEnabled(true);
            } catch (RuntimeException ignored) { louder=null; }
            byte[] waveform=null;
            try {
                visualizer=new Visualizer(player.getAudioSessionId());
                visualizer.setCaptureSize(Visualizer.getCaptureSizeRange()[1]);
                visualizer.setScalingMode(Visualizer.SCALING_MODE_AS_PLAYED);
                waveform=new byte[visualizer.getCaptureSize()];
                visualizer.setEnabled(true);
                lipSyncReady=true;
            } catch (RuntimeException ignored) { lipSyncReady=false; }
            synchronized (playbackLock) {
                if (item.terminal() || item.generation!=playbackGeneration) return;
                player.start();
                item.state="started";
                item.startedMs=SystemClock.elapsedRealtime();
                playbackError="";
                speech(0f,true);
            }
            while (!completed.get() && !failed.get()) {
                synchronized (playbackLock) {
                    if (item.terminal() || item.generation!=playbackGeneration) return;
                }
                float level=0f;
                if (waveform!=null && visualizer.getWaveForm(waveform)==Visualizer.SUCCESS) {
                    double sum=0;
                    for (byte sample:waveform) {
                        double value=((sample&255)-128)/128.0;
                        sum+=value*value;
                    }
                    level=(float)Math.max(0.,Math.min(1.,(Math.sqrt(sum/waveform.length)-.012)*8.));
                }
                synchronized (playbackLock) {
                    if (!item.terminal() && item.generation==playbackGeneration) speech(level,true);
                }
                SystemClock.sleep(25);
            }
            synchronized (playbackLock) {
                if (!item.terminal() && item.generation==playbackGeneration) {
                    item.state=completed.get() ? "completed" : "failed";
                    item.error=failed.get() ? "decoder_or_output_error" : "";
                    item.terminalMs=SystemClock.elapsedRealtime();
                    lastPlaybackMs=item.terminalMs;
                }
            }
        } catch (Exception error) {
            synchronized (playbackLock) {
                if (!item.terminal()) {
                    item.state="failed";
                    item.error="playback:"+error.getClass().getSimpleName();
                    item.terminalMs=SystemClock.elapsedRealtime();
                    playbackError=item.error;
                }
            }
        } finally {
            playbackWatchdog.removeCallbacks(expire);
            if (visualizer!=null) try { visualizer.release(); } catch (RuntimeException ignored) { }
            if (louder!=null) try { louder.release(); } catch (RuntimeException ignored) { }
            if (player!=null) {
                try { player.stop(); } catch (RuntimeException ignored) { }
                try { player.release(); } catch (RuntimeException ignored) { }
            }
            synchronized (playbackLock) {
                if (activeCompressedPlayer==player) activeCompressedPlayer=null;
                if (playbackThread==Thread.currentThread()) playbackThread=null;
                speech(0f,false);
            }
        }
    }

    private static final class ByteArrayMediaDataSource extends MediaDataSource {
        private final byte[] data;

        ByteArrayMediaDataSource(byte[] data) { this.data = data; }

        @Override public int readAt(long position, byte[] buffer, int offset, int size) {
            if (position < 0 || position >= data.length) return -1;
            int count = Math.min(size, data.length - (int)position);
            System.arraycopy(data, (int)position, buffer, offset, count);
            return count;
        }

        @Override public long getSize() { return data.length; }
        @Override public void close() { }
    }

    String healthJson() {
        long age = lastAudioMs == 0 ? -1 : SystemClock.elapsedRealtime() - lastAudioMs;
        long playbackAge = lastPlaybackMs == 0 ? -1 : SystemClock.elapsedRealtime() - lastPlaybackMs;
        return "{\"ready\":" + isReady() + ",\"service\":\"robotbrain-audio\"," +
                "\"producer_instance\":\""+producerInstance+"\",\"capture_generation\":"+captureGeneration+","+
                "\"capture_running\":"+running+",\"capture_retries\":"+captureRetries+","+
                "\"microphone\":\"front_phone_array\",\"sample_rate_hz\":" + SAMPLE_RATE + "," +
                "\"channels\":1,\"encoding\":\"pcm_s16le\",\"window_seconds\":" + WINDOW_SECONDS + "," +
                "\"noise_suppression_enabled\":" + noiseSuppressionEnabled + "," +
                "\"automatic_gain_enabled\":" + automaticGainEnabled + "," +
                "\"echo_cancellation_enabled\":" + echoCancellationEnabled + "," +
                "\"rms\":" + String.format(java.util.Locale.US, "%.6f", rms) + ",\"frame_age_ms\":" + age + "," +
                "\"playing\":" + playing + ",\"lip_sync_ready\":" + lipSyncReady + "," +
                "\"speech_level\":" + speechLevel + ",\"playback_error\":\"" + json(playbackError) + "\"," +
                "\"last_playback_age_ms\":" + playbackAge + ",\"motion_api\":false," +
                "\"last_error\":\"" + json(lastError) + "\"}";
    }

    void shutdown() {
        stopCapture();
        liveAudio.shutdown();
        synchronized (playbackLock) {
            for (Playback item: playbacks.values()) {
                if (!item.terminal()) cancelPlayback(item.id);
            }
        }
    }

    private static byte[] wav(byte[] pcm, int rate, int channels, int bits) {
        ByteBuffer output = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
        output.putInt(0x46464952).putInt(36 + pcm.length).putInt(0x45564157);
        output.putInt(0x20746d66).putInt(16).putShort((short)1).putShort((short)channels);
        output.putInt(rate).putInt(rate * channels * bits / 8).putShort((short)(channels * bits / 8)).putShort((short)bits);
        output.putInt(0x61746164).putInt(pcm.length).put(pcm);
        return output.array();
    }

    private static String safe(String value) { return value == null ? "" : value; }
    private static String json(String value) {
        return safe(value).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", " ");
    }
}
