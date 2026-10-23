package ai.robotbrain.face;

import android.net.Uri;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.FilterInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Semaphore;

/** A deliberately tiny, localhost-only state gateway. It has no motion API. */
public final class FaceServer extends Thread {
    private final FaceView faceView;
    private final CameraCapture cameraCapture;
    private final AudioIO audioIO;
    private final ProxyBroker proxyBroker;
    private final BleMoodLight moodLight;
    private final Semaphore requestCapacity = new Semaphore(4);
    private final Map<String, RateWindow> rateWindows = new HashMap<>();
    private volatile boolean running = true;
    private ServerSocket server;

    FaceServer(FaceView faceView, CameraCapture cameraCapture, AudioIO audioIO,
               ProxyBroker proxyBroker, BleMoodLight moodLight) {
        super("robotbrain-face-local-api");
        this.faceView = faceView;
        this.cameraCapture = cameraCapture;
        this.audioIO = audioIO;
        this.proxyBroker = proxyBroker;
        this.moodLight = moodLight;
        setDaemon(true);
    }

    @Override public void run() {
        ServerSocket listener = null;
        try {
            listener = new ServerSocket(8766, 8, InetAddress.getByName("127.0.0.1"));
            synchronized (this) {
                if (!running) {
                    listener.close();
                    return;
                }
                server = listener;
            }
            while (running) {
                try {
                    Socket socket = listener.accept();
                    if (!socket.getInetAddress().isLoopbackAddress() ||
                            !requestCapacity.tryAcquire()) {
                        socket.close();
                        continue;
                    }
                    socket.setSoTimeout(750);
                    Thread worker = new Thread(() -> {
                        try (Socket owned = socket) {
                            handle(owned);
                        } catch (IOException ignored) {
                        } finally {
                            requestCapacity.release();
                        }
                    }, "robotbrain-face-request");
                    worker.setDaemon(true);
                    worker.start();
                } catch (IOException ignored) {
                    if (!running) return;
                }
            }
        } catch (IOException ignored) {
        } finally {
            if (listener != null) try { listener.close(); } catch (IOException ignored) { }
            synchronized (this) {
                if (server == listener) server = null;
            }
        }
    }

    private void handle(Socket socket) throws IOException {
        final long readDeadline = android.os.SystemClock.elapsedRealtime() + 15_000L;
        InputStream input = new FilterInputStream(new BufferedInputStream(socket.getInputStream())) {
            private void checkDeadline() throws IOException {
                if (android.os.SystemClock.elapsedRealtime() >= readDeadline)
                    throw new IOException("request_read_deadline");
            }
            @Override public int read() throws IOException {
                checkDeadline();
                int result = in.read();
                checkDeadline();
                return result;
            }
            @Override public int read(byte[] bytes, int offset, int length) throws IOException {
                checkDeadline();
                int result = in.read(bytes, offset, length);
                checkDeadline();
                return result;
            }
        };
        String request = readAsciiLine(input);
        if (request == null) return;
        String[] parts = request.split(" ", -1);
        if (parts.length != 3 ||
                !("HTTP/1.1".equals(parts[2]) || "HTTP/1.0".equals(parts[2]))) {
            writeJson(socket, 400, "{\"ok\":false,\"error\":\"request_line_invalid\"}");
            return;
        }
        String method = parts[0];
        String target = parts.length > 1 ? parts[1] : "/";
        String path = target;
        String query = "";
        int question = target.indexOf('?');
        if (question >= 0) {
            path = target.substring(0, question);
            query = target.substring(question + 1);
        }
        int contentLength = 0;
        String authorization = null;
        boolean browserRequest = false;
        boolean contentLengthSeen = false;
        int headerCount = 0;
        while (true) {
            String line = readAsciiLine(input);
            if (line == null || line.isEmpty()) break;
            if (++headerCount > 64) {
                writeJson(socket, 400, "{\"ok\":false,\"error\":\"too_many_headers\"}");
                return;
            }
            if (line.regionMatches(true, 0, "Content-Length:", 0, 15)) {
                if (contentLengthSeen) {
                    writeJson(socket, 400, "{\"ok\":false,\"error\":\"duplicate_content_length\"}");
                    return;
                }
                contentLengthSeen = true;
                try { contentLength = Integer.parseInt(line.substring(15).trim()); }
                catch (NumberFormatException ignored) { contentLength = -1; }
            } else if (line.regionMatches(true, 0, "Authorization:", 0, 14)) {
                if (authorization != null) {
                    writeJson(socket, 400, "{\"ok\":false,\"error\":\"duplicate_authorization\"}");
                    return;
                }
                authorization = line.substring(14).trim();
            } else if (line.regionMatches(true, 0, "Origin:", 0, 7) ||
                    line.regionMatches(true, 0, "Sec-Fetch-Site:", 0, 15)) {
                browserRequest = true;
            } else if (line.regionMatches(true, 0, "Transfer-Encoding:", 0, 18)) {
                writeJson(socket, 400, "{\"ok\":false,\"error\":\"transfer_encoding_forbidden\"}");
                return;
            }
        }
        if (contentLength < 0) {
            writeJson(socket, 400, "{\"ok\":false,\"error\":\"content_length_invalid\"}");
            return;
        }

        int status = 200;
        String body;
        byte[] binaryBody = null;
        String contentType = "application/json; charset=utf-8";
        String extraHeaders = "";
        String requiredScope = requiredScope(path);
        boolean detailedHealth = "/health".equals(path) && authorization != null;
        // Readback is authorized with the same local capability but gets its
        // own bucket, so diagnostics cannot starve real emotion writes.
        String rateScope = path.equals("/audio/live/append") ? "live.append" :
                path.equals("/audio/live/stop") ? null :
                path.startsWith("/audio/live/") ? "live.control" : "/state".equals(path) ? "face.state" :
                path.equals("/audio/playback") || path.equals("/audio/stop") ? "speaker.status" :
                requiredScope != null ? requiredScope :
                detailedHealth ? "status.read" : null;
        if ((requiredScope != null || detailedHealth ||
                "/proxy/public-key".equals(path) || "/proxy/provision".equals(path)) &&
                browserRequest) {
            status = 403;
            body = "{\"ok\":false,\"error\":\"browser_origin_forbidden\"}";
        } else if (requiredScope != null &&
                !proxyBroker.authorize(requiredScope, authorization)) {
            status = 401;
            body = "{\"ok\":false,\"error\":\"unauthorized\"}";
        } else if (detailedHealth && !proxyBroker.authorize("status.read", authorization)) {
            status = 401;
            body = "{\"ok\":false,\"error\":\"unauthorized\"}";
        } else if (rateScope != null && !withinRate(rateScope)) {
            status = 429;
            body = "{\"ok\":false,\"error\":\"rate_limited\"}";
        } else if ("/health".equals(path) && "GET".equals(method)) {
            boolean serviceReady = proxyBroker.ready() &&
                    cameraCapture.isReady() && audioIO.isReady();
            if (!detailedHealth) {
                body = "{\"ready\":" + serviceReady +
                        ",\"service\":\"robotbrain-face\",\"security_configured\":" +
                        proxyBroker.securityConfigured() + ",\"motion_api\":false}";
            } else {
                body = "{\"ready\":" + serviceReady +
                    ",\"service\":\"robotbrain-face\",\"state\":\"" +
                    faceView.getEmotionName() + "\",\"camera_ready\":" +
                    cameraCapture.isReady() + ",\"audio_ready\":" + audioIO.isReady() +
                    ",\"proxy_broker\":" + proxyBroker.healthJson() +
                    ",\"mood_light\":" + moodLight.healthJson() +
                    ",\"motion_api\":false}";
            }
        } else if ("/proxy/health".equals(path) && "GET".equals(method)) {
            body = proxyBroker.healthJson();
        } else if ("/proxy/public-key".equals(path) && "GET".equals(method)) {
            body = proxyBroker.publicKeyJson();
        } else if ("/proxy/provision".equals(path) && "POST".equals(method)) {
            if (contentLength <= 0 || contentLength > 128 * 1024) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid provision body size\"}";
            } else {
                ProxyBroker.ProvisionResult result = proxyBroker.provision(
                        readBody(input, contentLength));
                status = result.ok ? 200 : 400;
                body = result.json();
            }
        } else if ("/camera/health".equals(path) && "GET".equals(method)) {
            body = cameraCapture.healthJson();
            status = cameraCapture.isReady() ? 200 : 503;
        } else if ("/camera/config".equals(path) && "GET".equals(method)) {
            body = cameraCapture.configJson();
        } else if ("/audio/health".equals(path) && "GET".equals(method)) {
            body = audioIO.healthJson();
            status = audioIO.isReady() ? 200 : 503;
        } else if ("/audio/cursor".equals(path) && "GET".equals(method)) {
            status = audioIO.isReady() ? 200 : 503;
            body = "{\"cursor\":" + audioIO.cursor() + ",\"sample_rate_hz\":16000," +
                    "\"channels\":1,\"encoding\":\"pcm_s16le\",\"ready\":" + audioIO.isReady() +
                    ",\"producer_instance\":\""+audioIO.producerInstance()+"\",\"capture_generation\":"+
                    audioIO.captureGeneration()+"}";
        } else if ("/audio/stream.pcm".equals(path) && "GET".equals(method)) {
            long since = longParameter(query, "since", audioIO.cursor());
            int maximum = (int)Math.min(96000, Math.max(2,
                    longParameter(query, "max_bytes", 32000)));
            AudioIO.PcmChunk chunk = audioIO.readSince(since, maximum);
            status = audioIO.isReady() ? 200 : 503;
            body = null;
            binaryBody = chunk.pcm;
            contentType = "application/octet-stream";
            extraHeaders = "X-RobotBrain-Cursor-Start: " + chunk.startCursor + "\r\n" +
                    "X-RobotBrain-Cursor-End: " + chunk.endCursor + "\r\n" +
                    "X-RobotBrain-Cursor-Dropped: " + chunk.dropped + "\r\n" +
                    "X-RobotBrain-Producer-Instance: " + audioIO.producerInstance() + "\r\n" +
                    "X-RobotBrain-Capture-Generation: " + chunk.generation + "\r\n" +
                    "X-RobotBrain-Audio-Frame-Elapsed-Ms: " + chunk.frameElapsedMs + "\r\n" +
                    "X-RobotBrain-Audio-Frame-Age-Ms: " + Math.max(0L,android.os.SystemClock.elapsedRealtime()-chunk.frameElapsedMs)+"\r\n"+
                    "X-RobotBrain-Audio-Ready: " + audioIO.isReady() + "\r\n" +
                    "X-RobotBrain-Sample-Rate: 16000\r\n" +
                    "X-RobotBrain-Channels: 1\r\n";
        } else if ("/audio/microphone.wav".equals(path) && "GET".equals(method)) {
            byte[] wav = audioIO.latestWav();
            if (wav == null) {
                status = 503;
                body = "{\"ok\":false,\"error\":\"microphone audio unavailable\"}";
            } else {
                body = null;
                binaryBody = wav;
                contentType = "audio/wav";
                extraHeaders = "X-RobotBrain-Sample-Rate: 16000\r\n" +
                        "X-RobotBrain-Channels: 1\r\n";
            }
        } else if (path.startsWith("/audio/live/") && "POST".equals(method)) {
            String liveId=parameter(query,"playback_id");
            body=null;
            if(path.equals("/audio/live/start")) body=audioIO.startLive(liveId);
            else if(path.equals("/audio/live/stop")) body=audioIO.liveAudio.stop(liveId);
            else if(path.equals("/audio/live/status")) body=audioIO.liveAudio.status(liveId);
            else if(path.equals("/audio/live/append") && contentLength>0 && contentLength<=6400) {
                try {body=audioIO.liveAudio.append(liveId,Integer.parseInt(parameter(query,"sequence")),readBody(input,contentLength));}
                catch(NumberFormatException ignored) { }
            }
            if(body==null) {status=409;body="{\"ok\":false,\"error\":\"live_pcm_rejected\"}";}
        } else if ("/audio/play.wav".equals(path) && "POST".equals(method)) {
            if (contentLength <= 44 || contentLength > 8 * 1024 * 1024) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid WAV body size\"}";
            } else {
                byte[] wav = readBody(input, contentLength);
                body = audioIO.acceptPlayback(wav,parameter(query,"playback_id"),false);
                if (body == null) { status=409; body="{\"ok\":false,\"error\":\"playback_not_admitted\"}"; }
            }
        } else if ("/audio/play.mp3".equals(path) && "POST".equals(method)) {
            if (contentLength < 128 || contentLength > 8 * 1024 * 1024) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid MP3 body size\"}";
            } else {
                byte[] mp3 = readBody(input, contentLength);
                body = audioIO.acceptPlayback(mp3,parameter(query,"playback_id"),true);
                if (body == null) { status=409; body="{\"ok\":false,\"error\":\"playback_not_admitted\"}"; }
            }
        } else if ("/audio/playback".equals(path) && "GET".equals(method)) {
            body=audioIO.playbackJson(parameter(query,"playback_id"));
            if (body==null) { status=404; body="{\"ok\":false,\"error\":\"playback_unknown\"}"; }
        } else if ("/audio/stop".equals(path) && "POST".equals(method)) {
            body=audioIO.cancelPlayback(parameter(query,"playback_id"));
            if (body==null) { status=400; body="{\"ok\":false,\"error\":\"playback_cancel_rejected\"}"; }
        } else if ("/mood-light/preview".equals(path) && "POST".equals(method)) {
            int brightness = -1;
            try { brightness = Integer.parseInt(parameter(query, "brightness")); }
            catch (NumberFormatException ignored) { }
            boolean accepted = moodLight.previewBrightness(brightness);
            status = accepted ? 200 : 400;
            body = accepted ? "{\"ok\":true,\"color\":\"white\",\"brightness\":" +
                    brightness + ",\"ttl_s\":10}" :
                    "{\"ok\":false,\"error\":\"brightness must be 1..100 and BLE ready\"}";
        } else if ("/audio/effects".equals(path) && "POST".equals(method)) {
            String noise = parameter(query, "ns");
            String echo = parameter(query, "aec");
            body = audioIO.setEffects(noise.isEmpty() ? null : "1".equals(noise),
                    echo.isEmpty() ? null : "1".equals(echo));
        } else if ("/mood-light/effect".equals(path) && "POST".equals(method)) {
            String name = parameter(query, "name");
            int seconds = 4;
            try { seconds = Integer.parseInt(parameter(query, "seconds")); }
            catch (NumberFormatException ignored) { }
            boolean known = MoodLighting.effect(name) != null;
            boolean accepted = known && moodLight.showEffect(name, seconds);
            status = accepted ? 200 : known ? 409 : 400;
            body = accepted ? "{\"ok\":true,\"effect\":\"" + name.toLowerCase(Locale.US) + "\",\"seconds\":" +
                    seconds + "}" : known ? "{\"ok\":false,\"error\":\"light not ready or asleep\"}" :
                    "{\"ok\":false,\"error\":\"unknown effect\"}";
        } else if ("/display".equals(path) && "POST".equals(method)) {
            final boolean value = !"0".equals(parameter(query, "awake"));
            faceView.post(() -> faceView.setDisplayAwake(value));
            body = "{\"ok\":true,\"awake\":" + value + "}";
        } else if ("/camera/record".equals(path) && "POST".equals(method)) {
            cameraCapture.setRecording("1".equals(parameter(query, "on")));
            body = cameraCapture.recordJson();
        } else if ("/camera/record".equals(path) && "GET".equals(method)) {
            body = cameraCapture.recordJson();
        } else if ("/camera/rate".equals(path) && "POST".equals(method)) {
            try {
                cameraCapture.setRate(Double.parseDouble(parameter(query, "fps")));
                body = "{\"ok\":true,\"fps\":" + cameraCapture.rate() + "}";
            } catch (NumberFormatException invalid) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"fps\"}";
            }
        } else if ("/camera/frame.nv21".equals(path) &&
                ("GET".equals(method) || "HEAD".equals(method))) {
            CameraCapture.Frame frame = cameraCapture.latestFrame();
            if (frame == null) {
                status = 503;
                body = "{\"ok\":false,\"error\":\"front camera frame unavailable\"}";
            } else {
                body = null;
                binaryBody = frame.nv21;
                contentType = "application/x-robotbrain-nv21";
                extraHeaders = frameHeaders(frame) +
                        "X-RobotBrain-Pixel-Format: NV21\r\n";
            }
        } else if ("/camera/frame.jpg".equals(path) &&
                ("GET".equals(method) || "HEAD".equals(method))) {
            CameraCapture.Frame frame = cameraCapture.latestFrame();
            if (frame == null) {
                status = 503;
                body = "{\"ok\":false,\"error\":\"front camera frame unavailable\"}";
            } else {
                byte[] jpeg = frame.jpeg();
                if (jpeg == null) {
                    status = 503;
                    body = "{\"ok\":false,\"error\":\"JPEG compression failed\"}";
                } else {
                    body = null;
                    binaryBody = jpeg;
                    contentType = "image/jpeg";
                    extraHeaders = frameHeaders(frame);
                }
            }
        } else if ("/photo".equals(path) && "POST".equals(method)) {
            if (contentLength < 128 || contentLength > 3 * 1024 * 1024) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid photo size\"}";
            } else {
                byte[] jpeg = readBody(input, contentLength);
                long seconds = 8L;
                try { seconds = Long.parseLong(parameter(query, "seconds")); }
                catch (NumberFormatException ignored) { }
                android.graphics.BitmapFactory.Options bounds = new android.graphics.BitmapFactory.Options();
                bounds.inJustDecodeBounds = true;
                android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length, bounds);
                android.graphics.BitmapFactory.Options options = new android.graphics.BitmapFactory.Options();
                options.inSampleSize = 1;
                while (bounds.outWidth / (options.inSampleSize * 2) >= 1080 ||
                        bounds.outHeight / (options.inSampleSize * 2) >= 720) options.inSampleSize *= 2;
                android.graphics.Bitmap bitmap = bounds.outWidth > 0 ?
                        android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length, options) : null;
                if (bitmap == null) {
                    status = 400;
                    body = "{\"ok\":false,\"error\":\"photo_not_decodable\"}";
                } else {
                    final long durationMs = Math.max(1L, Math.min(30L, seconds)) * 1000L;
                    faceView.post(() -> faceView.showPhoto(bitmap, durationMs));
                    body = "{\"ok\":true,\"width\":" + bitmap.getWidth() + ",\"height\":" +
                            bitmap.getHeight() + ",\"seconds\":" + durationMs / 1000L + "}";
                }
            }
        } else if ("/photo/clear".equals(path) && "POST".equals(method)) {
            faceView.post(faceView::clearPhoto);
            body = "{\"ok\":true}";
        } else if ("/photo".equals(path) && "GET".equals(method)) {
            body = faceView.photoStateJson();
        } else if ("/state".equals(path) && query.isEmpty() && "GET".equals(method)) {
            body = faceView.stateAsJson();
        } else if ("/gaze".equals(path) && "POST".equals(method)) {
            try {
                final float x = Float.parseFloat(parameter(query, "x"));
                final float y = Float.parseFloat(parameter(query, "y"));
                if (!Float.isFinite(x) || !Float.isFinite(y) || Math.abs(x)>1f || Math.abs(y)>1f)
                    throw new IllegalArgumentException();
                CountDownLatch applied = new CountDownLatch(1);
                final long deadline = android.os.SystemClock.uptimeMillis()+700L;
                java.util.concurrent.atomic.AtomicBoolean done = new java.util.concurrent.atomic.AtomicBoolean();
                faceView.post(() -> {
                    if (android.os.SystemClock.uptimeMillis() <= deadline) {
                        faceView.setTargetGaze(x, y);
                        done.set(true);
                    }
                    applied.countDown();
                });
                if (!applied.await(700, TimeUnit.MILLISECONDS) || !done.get()) {
                    status = 503;
                    body = "{\"ok\":false,\"error\":\"gaze_timeout\"}";
                } else body = "{\"ok\":true}";
            } catch (IllegalArgumentException invalid) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid_gaze\"}";
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                status = 503;
                body = "{\"ok\":false}";
            }
        } else if ("/emotion".equals(path) && "POST".equals(method)) {
            String requested = parameter(query, "state");
            String source = parameter(query, "source");
            String eventId = parameter(query, "event_id");
            String intensityValue = parameter(query, "intensity");
            String dockedValue = parameter(query, "docked");
            boolean dockedValid = dockedValue.isEmpty() ||
                    (("body".equals(source)) &&
                            ("0".equals(dockedValue) || "1".equals(dockedValue)));
            float intensity = 1f;
            boolean intensityValid = true;
            if (!intensityValue.isEmpty()) {
                try {
                    intensity = Float.parseFloat(intensityValue);
                    intensityValid = Float.isFinite(intensity) &&
                            intensity >= 0f && intensity <= 1f;
                } catch (NumberFormatException invalid) {
                    intensityValid = false;
                }
            }
            if (!FaceView.isKnownEmotion(requested)) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"unknown state\",\"allowed\":" +
                        FaceView.allowedAsJson() + "}";
            } else if (!source.isEmpty() && !"voice".equals(source) &&
                    !"body".equals(source) && !"event".equals(source) &&
                    !"reaction".equals(source) &&
                    !"direct".equals(source)) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"unknown emotion source\"}";
            } else if (eventId.length() > 96 ||
                    !eventId.matches("[A-Za-z0-9._:-]*")) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid emotion event id\"}";
            } else if (!intensityValid) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid emotion intensity\"}";
            } else if (!dockedValid) {
                status = 400;
                body = "{\"ok\":false,\"error\":\"invalid docked state\"}";
            } else {
                if (!applyEmotionAndWait(requested, source, eventId, intensity)) {
                    status = 503;
                    body = "{\"ok\":false,\"error\":\"face_apply_timeout\"}";
                } else {
                    if (!dockedValue.isEmpty()) {
                        moodLight.setDocked("1".equals(dockedValue));
                    }
                    body = "{\"ok\":true,\"state\":\"" +
                            requested.toLowerCase(Locale.US) +
                            "\",\"source\":\"" +
                            (source.isEmpty() ? "direct" : source) +
                            "\",\"event_id\":\"" + eventId +
                            "\",\"intensity\":" +
                            String.format(Locale.US, "%.3f", intensity) +
                            ",\"layers\":" + faceView.stateAsJson() + "}";
                }
            }
        } else if (isKnownPath(path)) {
            status = 405;
            body = "{\"ok\":false,\"error\":\"method_not_allowed\"}";
        } else {
            status = 404;
            body = "{\"ok\":false,\"error\":\"not found\"}";
        }
        byte[] bytes = binaryBody != null ? binaryBody : body.getBytes(StandardCharsets.UTF_8);
        String reason = reason(status);
        String head = "HTTP/1.1 " + status + reason + "\r\n" +
                "Content-Type: " + contentType + "\r\n" + extraHeaders +
                "Cache-Control: no-store\r\n" +
                "X-Content-Type-Options: nosniff\r\n" +
                "Content-Length: " + bytes.length + "\r\n" +
                "Connection: close\r\n\r\n";
        OutputStream output = socket.getOutputStream();
        output.write(head.getBytes(StandardCharsets.US_ASCII));
        if (!"HEAD".equals(method)) output.write(bytes);
        output.flush();
    }

    private static String requiredScope(String path) {
        if (path.startsWith("/camera/")) return "camera.read";
        if (path.equals("/audio/health") || path.equals("/audio/cursor") ||
                path.equals("/audio/stream.pcm") ||
                path.equals("/audio/microphone.wav")) return "microphone.read";
        if (path.startsWith("/audio/live/")) return "speaker.write";
        if (path.equals("/audio/play.wav") || path.equals("/audio/play.mp3") ||
                path.equals("/audio/playback") || path.equals("/audio/stop")) {
            return "speaker.write";
        }
        if (path.equals("/mood-light/preview") || path.equals("/mood-light/effect") ||
                path.equals("/audio/effects")) return "face.write";
        if (path.equals("/gaze") || path.equals("/emotion") || path.equals("/state") ||
                path.equals("/photo") || path.equals("/photo/clear") || path.equals("/display")) return "face.write";
        if (path.equals("/proxy/health")) return "proxy.read";
        return null;
    }

    private boolean applyEmotionAndWait(String state, String source,
                                        String eventId, float intensity) {
        CountDownLatch applied = new CountDownLatch(1);
        final long deadline=android.os.SystemClock.elapsedRealtime()+750L;
        java.util.concurrent.atomic.AtomicBoolean cancelled=new java.util.concurrent.atomic.AtomicBoolean();
        java.util.concurrent.atomic.AtomicBoolean succeeded=new java.util.concurrent.atomic.AtomicBoolean();
        if (!faceView.post(() -> {
            try {
                if (cancelled.get() || android.os.SystemClock.elapsedRealtime()>deadline) return;
                faceView.setEmotion(state, source, eventId, intensity);
                succeeded.set(true);
            } catch (RuntimeException ignored) {
            } finally {
                applied.countDown();
            }
        })) return false;
        try {
            boolean finished=applied.await(750, TimeUnit.MILLISECONDS);
            if (!finished) cancelled.set(true);
            return finished && succeeded.get();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            cancelled.set(true);
            return false;
        }
    }

    private static boolean isKnownPath(String path) {
        return path.equals("/audio/live/start") || path.equals("/audio/live/append") ||
                path.equals("/audio/live/stop") || path.equals("/audio/live/status") ||
                path.equals("/health") || path.equals("/proxy/health") ||
                path.equals("/photo") || path.equals("/photo/clear") ||
                path.equals("/proxy/public-key") || path.equals("/proxy/provision") ||
                path.equals("/camera/health") || path.equals("/camera/config") ||
                path.equals("/camera/frame.jpg") || path.equals("/camera/frame.nv21") ||
                path.equals("/camera/rate") || path.equals("/camera/record") || path.equals("/display") ||
                path.equals("/audio/health") ||
                path.equals("/audio/cursor") || path.equals("/audio/stream.pcm") ||
                path.equals("/audio/microphone.wav") ||
                path.equals("/audio/play.wav") || path.equals("/audio/play.mp3") ||
                path.equals("/audio/playback") || path.equals("/audio/stop") ||
                path.equals("/mood-light/preview") || path.equals("/mood-light/effect") ||
                path.equals("/audio/effects") ||
                path.equals("/gaze") || path.equals("/emotion") || path.equals("/state");
    }

    private synchronized boolean withinRate(String scope) {
        long now = android.os.SystemClock.elapsedRealtime();
        long windowMs = (scope.endsWith(".write") || "proxy.read".equals(scope)) ?
                60_000L : 10_000L;
        int maximum = "live.append".equals(scope) ? 180 : "speaker.write".equals(scope) ? 20 :
                "face.write".equals(scope) ? 120 :
                "proxy.read".equals(scope) ? 60 : 240;
        RateWindow window = rateWindows.get(scope);
        if (window == null || now - window.startedMs >= windowMs) {
            rateWindows.put(scope, new RateWindow(now, 1));
            return true;
        }
        if (window.count >= maximum) return false;
        window.count++;
        return true;
    }

    private static String frameHeaders(CameraCapture.Frame frame) {
        return "X-RobotBrain-Frame-Sequence: " + frame.sequence + "\r\n" +
                "X-RobotBrain-Camera-Generation: " + frame.generation + "\r\n" +
                "X-RobotBrain-Producer-Instance: " + frame.producerInstance + "\r\n" +
                "X-RobotBrain-Sensor-Timestamp-Ns: " + frame.sensorTimestampNs + "\r\n" +
                "X-RobotBrain-Frame-Width: " + frame.width + "\r\n" +
                "X-RobotBrain-Frame-Height: " + frame.height + "\r\n" +
                "X-RobotBrain-Camera-Id: 1\r\n" +
                "X-RobotBrain-Mirrored: false\r\n";
    }

    private static String reason(int status) {
        switch (status) {
            case 200: return " OK";
            case 400: return " Bad Request";
            case 401: return " Unauthorized";
            case 403: return " Forbidden";
            case 404: return " Not Found";
            case 405: return " Method Not Allowed";
            case 409: return " Conflict";
            case 429: return " Too Many Requests";
            case 503: return " Service Unavailable";
            default: return " Error";
        }
    }

    private static void writeJson(Socket socket, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        String head = "HTTP/1.1 " + status + reason(status) + "\r\n" +
                "Content-Type: application/json; charset=utf-8\r\n" +
                "Cache-Control: no-store\r\n" +
                "X-Content-Type-Options: nosniff\r\n" +
                "Content-Length: " + bytes.length + "\r\n" +
                "Connection: close\r\n\r\n";
        OutputStream output = socket.getOutputStream();
        output.write(head.getBytes(StandardCharsets.US_ASCII));
        output.write(bytes);
        output.flush();
    }

    private static final class RateWindow {
        final long startedMs;
        int count;

        RateWindow(long startedMs, int count) {
            this.startedMs = startedMs;
            this.count = count;
        }
    }

    private static byte[] readBody(InputStream input, int length) throws IOException {
        byte[] bytes = new byte[length];
        int offset = 0;
        while (offset < length) {
            int count = input.read(bytes, offset, length - offset);
            if (count < 0) throw new IOException("short request body");
            offset += count;
        }
        return bytes;
    }

    private static String readAsciiLine(InputStream input) throws IOException {
        ByteArrayOutputStream line = new ByteArrayOutputStream(128);
        while (true) {
            if (line.size() >= 8192) throw new IOException("header_line_too_long");
            int value = input.read();
            if (value < 0) return line.size() == 0 ? null : line.toString("US-ASCII");
            if (value == '\n') break;
            if (value != '\r') line.write(value);
        }
        return line.toString("US-ASCII");
    }

    private static String parameter(String query, String key) {
        for (String pair : query.split("&")) {
            int equals = pair.indexOf('=');
            if (equals > 0 && key.equals(pair.substring(0, equals))) {
                return Uri.decode(pair.substring(equals + 1));
            }
        }
        return "";
    }

    private static long longParameter(String query, String key, long fallback) {
        String value = parameter(query, key);
        if (value.isEmpty()) return fallback;
        try { return Long.parseLong(value); }
        catch (NumberFormatException ignored) { return fallback; }
    }

    synchronized void shutdown() {
        running = false;
        if (server != null) try { server.close(); } catch (IOException ignored) { }
    }
}
