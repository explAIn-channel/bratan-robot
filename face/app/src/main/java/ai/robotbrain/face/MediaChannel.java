package ai.robotbrain.face;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.media.audiofx.LoudnessEnhancer;
import android.os.SystemClock;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;

/**
 * One persistent localhost connection for the robot's voice: microphone frames out
 * as they are captured, speaker PCM in as it is generated, small controls.
 *
 * Client -> app: "AUTH <microphone.read token> <speaker.write token>\n", then frames
 *   [type u8][length u32 BE][payload]: 1 = PCM 16 kHz s16le mono to play,
 *   2 = stop playing now, 3 = control JSON ({"camera_fps": n}).
 * App -> client: 10 = [elapsedRealtimeNanos of first sample u64 BE][PCM],
 *   11 = status JSON {"queued_ms","playing","level"}.
 * A new client replaces the previous one. No network access, no motion.
 */
final class MediaChannel extends Thread {
    static final int PORT = 8767;
    private static final int RATE = 16000;
    private static final int MAX_QUEUE_BYTES = RATE * 2 * 60;
    private final AudioIO audio;
    private final CameraCapture camera;
    private final ProxyBroker broker;
    private volatile boolean running = true;
    private volatile Socket current;
    private final Object queueLock = new Object();
    private final ArrayDeque<byte[]> queue = new ArrayDeque<>();
    private int queuedBytes;
    private volatile boolean playing;
    private volatile float level;
    private Thread player;

    MediaChannel(AudioIO audio, CameraCapture camera, ProxyBroker broker) {
        super("robotbrain-media");
        this.audio = audio;
        this.camera = camera;
        this.broker = broker;
        setDaemon(true);
    }

    @Override public void run() {
        player = new Thread(this::playLoop, "robotbrain-media-speaker");
        player.setDaemon(true);
        player.start();
        try (ServerSocket server = new ServerSocket(PORT, 4, InetAddress.getByName("127.0.0.1"))) {
            while (running) {
                Socket socket = server.accept();
                if (!socket.getInetAddress().isLoopbackAddress()) { socket.close(); continue; }
                Socket previous = current;
                current = socket;
                if (previous != null) try { previous.close(); } catch (IOException ignored) { }
                Thread worker = new Thread(() -> serve(socket), "robotbrain-media-client");
                worker.setDaemon(true);
                worker.start();
            }
        } catch (IOException ignored) {
        }
    }

    void shutdown() {
        running = false;
        Socket socket = current;
        if (socket != null) try { socket.close(); } catch (IOException ignored) { }
        flush();
    }

    private void serve(Socket socket) {
        try {
            socket.setTcpNoDelay(true);
            socket.setSoTimeout(5000);
            InputStream raw = new BufferedInputStream(socket.getInputStream());
            String line = readLine(raw);
            String[] parts = line == null ? new String[0] : line.split(" ");
            if (parts.length != 3 || !"AUTH".equals(parts[0]) ||
                    !broker.authorize("microphone.read", "Bearer " + parts[1]) ||
                    !broker.authorize("speaker.write", "Bearer " + parts[2])) {
                socket.close();
                return;
            }
            socket.setSoTimeout(0);
            OutputStream out = socket.getOutputStream();
            Thread sender = new Thread(() -> sendLoop(socket, out), "robotbrain-media-mic");
            sender.setDaemon(true);
            sender.start();
            DataInputStream in = new DataInputStream(raw);
            while (running && current == socket) {
                int type = in.readUnsignedByte();
                int length = in.readInt();
                if (length < 0 || length > 1 << 20) break;
                byte[] payload = new byte[length];
                in.readFully(payload);
                if (type == 1) enqueue(payload);
                else if (type == 2) flush();
                else if (type == 3) control(new String(payload, StandardCharsets.UTF_8));
            }
        } catch (IOException ignored) {
        } finally {
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private void sendLoop(Socket socket, OutputStream out) {
        long cursor = audio.cursor();
        long lastStatus = 0;
        try {
            while (running && current == socket && !socket.isClosed()) {
                AudioIO.PcmChunk chunk = audio.readSince(cursor, 6400);
                if (chunk.dropped) cursor = chunk.endCursor;
                if (chunk.pcm.length > 0) {
                    cursor = chunk.endCursor;
                    long firstNs = chunk.frameElapsedMs * 1_000_000L - chunk.pcm.length * 1_000_000_000L / (RATE * 2);
                    ByteBuffer header = ByteBuffer.allocate(13);
                    header.put((byte) 10).putInt(8 + chunk.pcm.length).putLong(firstNs);
                    synchronized (out) {
                        out.write(header.array());
                        out.write(chunk.pcm);
                        out.flush();
                    }
                }
                long now = SystemClock.elapsedRealtime();
                if (now - lastStatus >= 100) {
                    lastStatus = now;
                    int queued;
                    synchronized (queueLock) { queued = queuedBytes; }
                    byte[] json = ("{\"queued_ms\":" + (queued * 1000L / (RATE * 2)) + ",\"playing\":" + playing +
                            ",\"level\":" + String.format(java.util.Locale.US, "%.3f", level) + "}")
                            .getBytes(StandardCharsets.UTF_8);
                    ByteBuffer header = ByteBuffer.allocate(5);
                    header.put((byte) 11).putInt(json.length);
                    synchronized (out) {
                        out.write(header.array());
                        out.write(json);
                        out.flush();
                    }
                }
                SystemClock.sleep(10);
            }
        } catch (IOException ignored) {
        } finally {
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private void control(String json) {
        try {
            org.json.JSONObject value = new org.json.JSONObject(json);
            if (value.has("camera_fps")) camera.setRate(value.getDouble("camera_fps"));
        } catch (Exception ignored) {
        }
    }

    private void enqueue(byte[] pcm) {
        if (pcm.length == 0 || (pcm.length & 1) != 0) return;
        synchronized (queueLock) {
            if (queuedBytes + pcm.length > MAX_QUEUE_BYTES) return;
            queue.add(pcm);
            queuedBytes += pcm.length;
            queueLock.notifyAll();
        }
    }

    private void flush() {
        synchronized (queueLock) {
            queue.clear();
            queuedBytes = 0;
            queueLock.notifyAll();
        }
        flushTrack = true;
    }

    private volatile boolean flushTrack;

    private void playLoop() {
        AudioTrack track = null;
        LoudnessEnhancer louder = null;
        long idleSince = SystemClock.elapsedRealtime();
        while (running) {
            byte[] pcm;
            synchronized (queueLock) {
                if (queue.isEmpty()) {
                    try { queueLock.wait(100); } catch (InterruptedException interrupted) { return; }
                }
                pcm = queue.poll();
                if (pcm != null) queuedBytes -= pcm.length;
            }
            if (flushTrack && track != null) {
                flushTrack = false;
                try { track.pause(); track.flush(); track.play(); } catch (RuntimeException ignored) { }
            }
            if (pcm == null) {
                if (playing) { playing = false; level = 0; audio.meter(0, false); }
                // Release the track after a quiet minute; keep it warm between sentences.
                if (track != null && SystemClock.elapsedRealtime() - idleSince > 60_000) {
                    if (louder != null) try { louder.release(); } catch (RuntimeException ignored) { }
                    try { track.release(); } catch (RuntimeException ignored) { }
                    track = null;
                    louder = null;
                }
                continue;
            }
            idleSince = SystemClock.elapsedRealtime();
            try {
                if (track == null) {
                    track = new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                            .setAudioFormat(new AudioFormat.Builder().setSampleRate(RATE)
                                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                            .setTransferMode(AudioTrack.MODE_STREAM)
                            .setBufferSizeInBytes(Math.max(6400, AudioTrack.getMinBufferSize(RATE,
                                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)))
                            .build();
                    try {
                        louder = new LoudnessEnhancer(track.getAudioSessionId());
                        louder.setTargetGain(AudioIO.SPEECH_BOOST_MB);
                        louder.setEnabled(true);
                    } catch (RuntimeException ignored) { louder = null; }
                    track.play();
                }
                long energy = 0;
                for (int i = 0; i + 1 < pcm.length; i += 2) {
                    int v = (short) ((pcm[i] & 255) | (pcm[i + 1] << 8));
                    energy += (long) v * v;
                }
                level = (float) Math.min(1, Math.sqrt(energy / (double) Math.max(1, pcm.length / 2)) / 5000.0);
                playing = true;
                audio.meter(level, level > 0.005f);
                int offset = 0;
                while (running && offset < pcm.length && !flushTrack) {
                    int n = track.write(pcm, offset, pcm.length - offset, AudioTrack.WRITE_BLOCKING);
                    if (n <= 0) throw new IllegalStateException("pcm_write_failed");
                    offset += n;
                }
            } catch (RuntimeException error) {
                if (track != null) try { track.release(); } catch (RuntimeException ignored) { }
                track = null;
                louder = null;
            }
        }
    }

    private static String readLine(InputStream in) throws IOException {
        StringBuilder line = new StringBuilder();
        for (int i = 0; i < 512; i++) {
            int c = in.read();
            if (c < 0) return null;
            if (c == '\n') return line.toString().trim();
            line.append((char) c);
        }
        return null;
    }
}
