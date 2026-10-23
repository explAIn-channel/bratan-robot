package ai.robotbrain.face;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Range;
import android.util.Size;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Arrays;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Locale;

/**
 * Front Camera2 capture for perception. Frames stay sensor-native and are never
 * selfie-mirrored; the ROS bridge applies the configured 270 degree rotation.
 */
public final class CameraCapture {
    public static final class Frame {
        public final byte[] nv21;
        public final int width;
        public final int height;
        public final long sequence;
        public final long sensorTimestampNs;
        public final long receivedElapsedMs;
        public final long generation;
        public final String producerInstance;
        private volatile byte[] jpeg;

        Frame(byte[] nv21, int width, int height, long sequence,
              long sensorTimestampNs, long receivedElapsedMs, long generation, String producerInstance) {
            this.nv21 = nv21;
            this.width = width;
            this.height = height;
            this.sequence = sequence;
            this.sensorTimestampNs = sensorTimestampNs;
            this.receivedElapsedMs = receivedElapsedMs;
            this.generation = generation;
            this.producerInstance = producerInstance;
        }

        /**
         * Preserve the public JPEG endpoint without paying its native encoder
         * cost for every camera frame. The continuous ROS consumer reads NV21;
         * occasional diagnostic/perception callers encode at most once for the
         * particular immutable frame they obtained.
         */
        synchronized byte[] jpeg() {
            if (jpeg != null) return jpeg;
            ByteArrayOutputStream output = new ByteArrayOutputStream(width * height / 3);
            YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
            if (!yuv.compressToJpeg(new Rect(0, 0, width, height), 84, output)) return null;
            jpeg = output.toByteArray();
            return jpeg;
        }
    }

    private final Context context;
    private final String cameraId;
    private final int requestedWidth;
    private final int requestedHeight;
    private volatile int maxProcessingFps;
    private final int sensorFps;
    private final Object lock = new Object();
    private HandlerThread thread;
    private Handler handler;
    private ImageReader imageReader;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private CaptureRequest captureRequest;
    private Frame latest;
    private long sequence;
    private long lastProcessedMs;
    private volatile boolean started;
    private volatile long generation;
    private final String producerInstance = java.util.UUID.randomUUID().toString();
    private volatile boolean captureActive;
    private volatile String lastError = "not started";
    private int sensorOrientation = 270;
    private int captureWidth;
    private int captureHeight;
    // TEMPLATE_PREVIEW leaves the sensor free-running at its full preview rate.
    // onImageAvailable then throttles *processing* to maxProcessingFps, but the
    // HAL, ISP and buffer pipeline still ran flat out: the camera provider and
    // this app together held roughly a whole core continuously, which starved
    // the sensor bridge badly enough that LiDAR publication fell from 5.3 Hz to
    // under 1 Hz. Constrain the sensor itself so the frames are never produced.
    private Range<Integer> targetFpsRange;
    // The trip video's range: the same freedom in the dark (face 90), at most 30 frames a second.
    private Range<Integer> recordFpsRange;
    // What the sensor really did on the last frame (face 90: to see the low-light range at work).
    private volatile long lastExposureNs, lastFrameDurationNs;
    private volatile int lastIso;
    private final CameraCaptureSession.CaptureCallback exposureWatch = new CameraCaptureSession.CaptureCallback() {
        @Override public void onCaptureCompleted(CameraCaptureSession s, CaptureRequest r, TotalCaptureResult result) {
            Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
            Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
            Long frame = result.get(CaptureResult.SENSOR_FRAME_DURATION);
            if (exposure != null) lastExposureNs = exposure;
            if (iso != null) lastIso = iso;
            if (frame != null) lastFrameDurationNs = frame;
        }
    };
    private Runnable captureTick;
    private Runnable captureTickFor(final long ownedGeneration) { return new Runnable() {
        @Override public void run() {
            CameraCaptureSession activeSession;
            CaptureRequest activeRequest;
            Handler activeHandler;
            synchronized (CameraCapture.this) {
                if (!started || !captureActive || generation != ownedGeneration) return;
                activeSession = session;
                activeRequest = captureRequest;
                activeHandler = handler;
            }
            if (activeSession == null || activeRequest == null || activeHandler == null) return;
            if (paused || recorder != null) {
                activeHandler.postDelayed(this, 500L);
                return;
            }
            try {
                // A repeating request makes this camera's HAL produce its
                // hardware minimum of 15 FPS even though RobotBrain consumes
                // only four. One bounded capture per tick keeps the camera
                // warm and fresh without manufacturing eleven discarded
                // frames every second.
                activeSession.capture(activeRequest, exposureWatch, activeHandler);
            } catch (CameraAccessException | IllegalStateException error) {
                synchronized (CameraCapture.this) {
                    if (generation==ownedGeneration) lastError = "capture tick: " + safe(error.getMessage());
                }
            }
            synchronized (CameraCapture.this) {
                if (started && captureActive && handler != null && generation == ownedGeneration) {
                    long period = slowPeriodMs > 0 ? slowPeriodMs : 1000L / maxProcessingFps;
                    handler.postDelayed(this, Math.max(1L, paused ? 500L : period));
                }
            }
        }
    }; }

    CameraCapture(Context context, String cameraId, int width, int height, int maxProcessingFps) {
        this.context = context.getApplicationContext();
        this.cameraId = cameraId;
        this.requestedWidth = width;
        this.requestedHeight = height;
        this.maxProcessingFps = Math.max(1, maxProcessingFps);
        this.sensorFps = 15;
    }

    @SuppressLint("MissingPermission")
    synchronized void start() {
        if (started) return;
        final long ownedGeneration = ++generation;
        synchronized (lock) { latest=null; sequence=0; lastProcessedMs=0; }
        captureTick=captureTickFor(ownedGeneration);
        started = true;
        lastError = "opening";
        thread = new HandlerThread("robotbrain-front-camera");
        thread.start();
        handler = new Handler(thread.getLooper());
        try {
            CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
            if (manager == null) throw new CameraAccessException(CameraAccessException.CAMERA_ERROR, "camera manager unavailable");
            CameraCharacteristics chars = manager.getCameraCharacteristics(cameraId);
            Integer facing = chars.get(CameraCharacteristics.LENS_FACING);
            if (facing == null || facing != CameraCharacteristics.LENS_FACING_FRONT) {
                throw new CameraAccessException(CameraAccessException.CAMERA_ERROR, "Camera ID " + cameraId + " is not front-facing");
            }
            Integer orientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION);
            if (orientation != null) sensorOrientation = orientation;
            targetFpsRange = selectFpsRange(chars, sensorFps);
            recordFpsRange = selectFpsRange(chars, TripRecorder.FPS);
            Size selected = selectSize(chars, requestedWidth, requestedHeight);
            captureWidth = selected.getWidth();
            captureHeight = selected.getHeight();
            imageReader = ImageReader.newInstance(captureWidth, captureHeight, ImageFormat.YUV_420_888, 3);
            final ImageReader ownedReader=imageReader;
            final Handler ownedHandler=handler;
            imageReader.setOnImageAvailableListener(reader -> onImageAvailable(reader,ownedGeneration), handler);
            manager.openCamera(cameraId, stateCallbackFor(ownedGeneration,ownedReader,ownedHandler), handler);
        } catch (Exception error) {
            lastError = error.getClass().getSimpleName() + ": " + safe(error.getMessage());
            handler.post(this::shutdown);
        }
    }

    // ---------------------------------------------------------------- trip recording
    private TripRecorder recorder;
    private AudioIO microphone;
    private volatile String lastTrip = "";
    private volatile String recordError = "";

    void setMicrophone(AudioIO audio) { this.microphone = audio; }

    File tripFolder() { return context.getExternalFilesDir("trips"); }

    /** Start/stop 720p30 H.264+AAC recording; the perception stream keeps flowing meanwhile. */
    void setRecording(final boolean on) {
        Handler owned;
        synchronized (this) { owned = handler; }
        if (owned == null) return;
        owned.post(() -> {
            if (on && recorder == null) startRecording();
            else if (!on && recorder != null) stopRecording();
        });
    }

    boolean recording() { return recorder != null; }

    String recordJson() {
        return "{\"recording\":" + (recorder != null) + ",\"last\":\"" + json(lastTrip) +
                "\",\"error\":\"" + json(recordError) + "\"}";
    }

    private void startRecording() {
        CameraDevice owned;
        ImageReader reader;
        synchronized (this) { owned = camera; reader = imageReader; }
        if (owned == null || reader == null || microphone == null) { recordError = "camera_not_ready"; return; }
        try {
            Size size = recordSize();
            recorder = new TripRecorder(tripFolder(), size.getWidth(), size.getHeight(), microphone);
            final TripRecorder ownedRecorder = recorder;
            final long ownedGeneration = generation;
            synchronized (this) {
                if (session != null) { try { session.close(); } catch (Exception ignored) { } session = null; }
                captureActive = false;
            }
            owned.createCaptureSession(Arrays.asList(reader.getSurface(), ownedRecorder.surface()),
                new CameraCaptureSession.StateCallback() {
                    @Override public void onConfigured(CameraCaptureSession configured) {
                        synchronized (CameraCapture.this) {
                            if (generation != ownedGeneration || recorder != ownedRecorder) { configured.close(); return; }
                            session = configured;
                            try {
                                CaptureRequest.Builder request = owned.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
                                request.addTarget(reader.getSurface());
                                request.addTarget(ownedRecorder.surface());
                                request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, recordFpsRange != null
                                        ? recordFpsRange : new Range<>(TripRecorder.FPS, TripRecorder.FPS));
                                configured.setRepeatingRequest(request.build(), exposureWatch, handler);
                                captureActive = true;
                                recordError = "";
                            } catch (CameraAccessException | IllegalStateException error) {
                                recordError = "record request: " + safe(error.getMessage());
                            }
                        }
                    }
                    @Override public void onConfigureFailed(CameraCaptureSession failed) {
                        failed.close();
                        recordError = "record session failed";
                        handler.post(CameraCapture.this::stopRecording);
                    }
                }, handler);
        } catch (Exception error) {
            recordError = error.getClass().getSimpleName() + ": " + safe(error.getMessage());
            if (recorder != null) { recorder.stop(); recorder = null; }
            reopenPerceptionSession();
        }
    }

    private void stopRecording() {
        TripRecorder owned = recorder;
        recorder = null;
        synchronized (this) {
            if (session != null) {
                try { session.stopRepeating(); } catch (Exception ignored) { }
                try { session.close(); } catch (Exception ignored) { }
                session = null;
            }
            captureActive = false;
        }
        if (owned != null) {
            File done = owned.stop();
            if (done != null) lastTrip = done.getName();
        }
        reopenPerceptionSession();
    }

    private void reopenPerceptionSession() {
        CameraDevice owned;
        ImageReader reader;
        Handler ownedHandler;
        synchronized (this) { owned = camera; reader = imageReader; ownedHandler = handler; }
        if (owned == null || reader == null || ownedHandler == null) return;
        try {
            owned.createCaptureSession(java.util.Collections.singletonList(reader.getSurface()),
                    sessionCallbackFor(generation, owned, reader, ownedHandler), ownedHandler);
        } catch (CameraAccessException error) {
            lastError = "reopen session: " + safe(error.getMessage());
        }
    }

    private Size recordSize() throws CameraAccessException {
        CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        android.hardware.camera2.params.StreamConfigurationMap map = manager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size[] sizes = map == null ? null : map.getOutputSizes(android.media.MediaCodec.class);
        Size best = new Size(TripRecorder.WIDTH, TripRecorder.HEIGHT);
        if (sizes == null) return best;
        long target = (long) TripRecorder.WIDTH * TripRecorder.HEIGHT, bestScore = Long.MAX_VALUE;
        for (Size s : sizes) {
            long area = (long) s.getWidth() * s.getHeight();
            if (area > target) continue;
            long score = Math.abs(area - target) + Math.abs((long) s.getWidth() * 9 - (long) s.getHeight() * 16) * 50L;
            if (score < bestScore) { bestScore = score; best = s; }
        }
        return best;
    }

    /** Frames produced per second (0.2..15); 0 pauses capture without closing the camera. */
    void setRate(double fps) {
        if (Double.isNaN(fps)) return;
        paused = fps < 0.1;
        maxProcessingFps = (int) Math.max(1, Math.min(15, Math.round(fps)));
        slowPeriodMs = fps < 1 && fps >= 0.1 ? (long) (1000 / fps) : 0;
    }

    private volatile boolean paused;
    private volatile long slowPeriodMs;

    int rate() { return paused ? 0 : maxProcessingFps; }

    void restart() {
        shutdown();
        start();
    }

    private Size selectSize(CameraCharacteristics chars, int desiredWidth, int desiredHeight) throws CameraAccessException {
        android.hardware.camera2.params.StreamConfigurationMap map =
                chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) throw new CameraAccessException(CameraAccessException.CAMERA_ERROR, "stream map unavailable");
        Size[] sizes = map.getOutputSizes(ImageFormat.YUV_420_888);
        if (sizes == null || sizes.length == 0) throw new CameraAccessException(CameraAccessException.CAMERA_ERROR, "no YUV output sizes");
        Size best = sizes[0];
        long desiredArea = (long) desiredWidth * desiredHeight;
        long bestScore = Long.MAX_VALUE;
        for (Size size : sizes) {
            long area = (long) size.getWidth() * size.getHeight();
            long ratioPenalty = Math.abs((long) size.getWidth() * desiredHeight - (long) size.getHeight() * desiredWidth) * 20L;
            long score = Math.abs(area - desiredArea) + ratioPenalty;
            if (score < bestScore) { best = size; bestScore = score; }
        }
        return best;
    }

    private CameraDevice.StateCallback stateCallbackFor(final long ownedGeneration,
            final ImageReader ownedReader, final Handler ownedHandler) { return new CameraDevice.StateCallback() {
        @Override public void onOpened(CameraDevice opened) {
            synchronized (CameraCapture.this) {
            if (!started || generation!=ownedGeneration) { opened.close(); return; }
            camera = opened;
            try {
                opened.createCaptureSession(Collections.singletonList(ownedReader.getSurface()),
                    sessionCallbackFor(ownedGeneration,opened,ownedReader,ownedHandler),ownedHandler);
            } catch (CameraAccessException error) {
                lastError = "create session: " + safe(error.getMessage());
            }
            }
        }
        @Override public void onDisconnected(CameraDevice disconnected) {
            synchronized (CameraCapture.this) {
            if (generation!=ownedGeneration) { disconnected.close(); return; }
            captureActive = false;
            lastError = "camera disconnected";
            disconnected.close();
            camera = null;
            }
        }
        @Override public void onError(CameraDevice failed, int error) {
            synchronized (CameraCapture.this) {
            if (generation!=ownedGeneration) { failed.close(); return; }
            captureActive = false;
            lastError = "camera error " + error;
            failed.close();
            camera = null;
            }
        }
    }; }

    private CameraCaptureSession.StateCallback sessionCallbackFor(final long ownedGeneration,
            final CameraDevice ownedCamera, final ImageReader ownedReader, final Handler ownedHandler) {
        return new CameraCaptureSession.StateCallback() {
        @Override public void onConfigured(CameraCaptureSession configured) {
            synchronized (CameraCapture.this) {
            if (!started || generation!=ownedGeneration || camera!=ownedCamera) { configured.close(); return; }
            session = configured;
            try {
                CaptureRequest.Builder request = ownedCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                request.addTarget(ownedReader.getSurface());
                request.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                request.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF);
                if (targetFpsRange != null) {
                    request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, targetFpsRange);
                }
                captureRequest = request.build();
                captureActive = true;
                lastError = "";
                ownedHandler.removeCallbacks(captureTick);
                ownedHandler.post(captureTick);
            } catch (CameraAccessException error) {
                lastError = "start capture: " + safe(error.getMessage());
            }
            }
        }
        @Override public void onConfigureFailed(CameraCaptureSession failed) {
            synchronized (CameraCapture.this) {
            failed.close();
            if (generation!=ownedGeneration) return;
            captureActive = false;
            lastError = "capture session configuration failed";
            }
        }
    }; }

    /**
     * Pick the cheapest sensor frame rate that still satisfies the processing
     * rate. Prefers a fixed range so the sensor cannot drift back up under
     * auto-exposure, and never selects a range slower than requested.
     */
    /**
     * The range that reaches {@code desiredFps} and lets auto-exposure slow down the most in the dark (face 90).
     * The fixed 15-15 taken before capped a frame at 1/15 s, the trip video's 30-30 at 1/30 s: in a dim room the
     * frames were nearly black (brightness 7-25 of 255) and nobody was seen. This front camera offers 15-15, 7-30
     * and 30-30; 7-30 lets a frame last up to 1/7 s. Perception frames are taken one request at a time, so a range
     * reaching 30 adds no frames; in good light the exposure stays as short as before.
     */
    static Range<Integer> selectFpsRange(CameraCharacteristics chars, int desiredFps) {
        Range<Integer>[] available =
                chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        if (available == null || available.length == 0) return null;
        Range<Integer> best = null;
        for (Range<Integer> candidate : available) {
            if (candidate == null || candidate.getUpper() < desiredFps) continue;
            if (best == null) { best = candidate; continue; }
            int byLower = Integer.compare(candidate.getLower(), best.getLower());
            if (byLower < 0 || (byLower == 0 && candidate.getUpper() < best.getUpper())) best = candidate;
        }
        return best;
    }

    private void onImageAvailable(ImageReader reader, long ownedGeneration) {
        try (Image image = reader.acquireLatestImage()) {
            if (image == null) return;
            long now = SystemClock.elapsedRealtime();
            if (!started || generation!=ownedGeneration) return;
            if (now - lastProcessedMs < (slowPeriodMs > 0 ? slowPeriodMs : 1000L / maxProcessingFps) - 15) return;
            lastProcessedMs = now;
            byte[] nv21 = toNv21(image);
            synchronized (CameraCapture.this) {
                if (!started || generation!=ownedGeneration) return;
                Frame frame = new Frame(nv21, image.getWidth(), image.getHeight(), ++sequence,
                    image.getTimestamp(), now,ownedGeneration,producerInstance);
                synchronized (lock) { latest = frame; }
            }
        } catch (Exception error) {
            if (generation!=ownedGeneration) return;
            lastError = "frame conversion: " + error.getClass().getSimpleName() + ": " + safe(error.getMessage());
        }
    }

    private static byte[] toNv21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] output = new byte[width * height * 3 / 2];
        Image.Plane[] planes = image.getPlanes();
        copyPlane(planes[0], width, height, output, 0, 1);
        int chromaWidth = width / 2;
        int chromaHeight = height / 2;
        copyPlane(planes[2], chromaWidth, chromaHeight, output, width * height, 2);
        copyPlane(planes[1], chromaWidth, chromaHeight, output, width * height + 1, 2);
        return output;
    }

    private static void copyPlane(Image.Plane plane, int width, int height, byte[] output, int offset, int outputStride) {
        ByteBuffer buffer = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        for (int row = 0; row < height; row++) {
            int rowStart = row * rowStride;
            for (int col = 0; col < width; col++) {
                output[offset + (row * width + col) * outputStride] = buffer.get(rowStart + col * pixelStride);
            }
        }
    }

    boolean isReady() {
        Frame frame;
        synchronized (lock) { frame = latest; }
        return started && captureActive && frame != null && frame.generation==generation &&
                SystemClock.elapsedRealtime() - frame.receivedElapsedMs < 2000;
    }

    boolean isCaptureActive() {
        return captureActive;
    }

    Frame latestFrame() {
        synchronized (lock) { return isReady() ? latest : null; }
    }

    String configJson() {
        return "{\"camera_id\":\"" + cameraId + "\",\"role\":\"front_navigation\"," +
                "\"facing\":\"front\",\"sensor_orientation_deg\":" + sensorOrientation + "," +
                "\"sensor_native_width\":" + captureWidth + ",\"sensor_native_height\":" + captureHeight + "," +
                "\"processing_rotation_clockwise_deg\":270,\"processing_mirror\":false," +
                "\"focus\":\"fixed\",\"max_processing_fps\":" + maxProcessingFps + "}";
    }

    String healthJson() {
        Frame frame;
        synchronized (lock) { frame = latest; }
        long age = frame == null ? -1 : SystemClock.elapsedRealtime() - frame.receivedElapsedMs;
        return "{\"ready\":" + isReady() + ",\"service\":\"robotbrain-front-camera\"," +
                "\"camera_id\":\"" + cameraId + "\",\"role\":\"front_navigation\"," +
                "\"frame_sequence\":" + (frame == null ? 0 : frame.sequence) + ",\"frame_age_ms\":" + age + "," +
                "\"mirrored\":false,\"motion_api\":false,\"last_error\":\"" + json(lastError) + "\"," +
                "\"fps_range\":\"" + targetFpsRange + "\",\"record_fps_range\":\"" + recordFpsRange + "\"," +
                "\"exposure_ms\":" + String.format(Locale.US, "%.1f", lastExposureNs / 1e6) + "," +
                "\"frame_ms\":" + String.format(Locale.US, "%.1f", lastFrameDurationNs / 1e6) + "," +
                "\"iso\":" + lastIso + "}";
    }


    void shutdown() {
        HandlerThread ownedThread;
        if (recorder != null) { TripRecorder owned = recorder; recorder = null; owned.stop(); }
        synchronized (this) {
            started=false;
            ++generation;
            captureActive=false;
            captureRequest=null;
            if (handler!=null) handler.removeCallbacksAndMessages(null);
            if (session!=null) { try { session.close(); } catch (Exception ignored) { } session=null; }
            if (camera!=null) { try { camera.close(); } catch (Exception ignored) { } camera=null; }
            if (imageReader!=null) { try { imageReader.close(); } catch (Exception ignored) { } imageReader=null; }
            synchronized (lock) { latest=null; }
            ownedThread=thread;
            thread=null;
            handler=null;
        }
        if (ownedThread!=null) {
            ownedThread.quitSafely();
            if (ownedThread!=Thread.currentThread()) {
                try { ownedThread.join(500); } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
            }
        }
    }

    private static String safe(String value) { return value == null ? "" : value; }
    private static String json(String value) {
        return safe(value).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", " ");
    }
}
