package ai.robotbrain.face;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;

import java.util.Locale;
import java.util.Random;

/**
 * Semantic state mixer for the undamaged top third of the Meizu display.
 * The production renderer is a complete corgi face whose expression grammar
 * is built from independent human-like facial actions.
 */
public final class FaceView extends View {
    private enum Emotion {
        IDLE, LISTENING, THINKING, SPEAKING, HAPPY, LAUGHING,
        CURIOUS, SURPRISED, AFFECTIONATE, SLEEPY, SAD,
        CONCERNED, ANNOYED, ERROR
    }

    private static final Emotion[] IDLE_MICRO_AFFECTS = {
            Emotion.HAPPY, Emotion.CURIOUS, Emotion.SURPRISED, Emotion.ANNOYED
    };
    private static final boolean IDLE_MICRO_MOODS = false;
    private static final int AMBIENT_VISUAL_EXIT_MS = 190;

    private static final String[] NAMES = {
            "idle", "listening", "thinking", "speaking", "happy", "laughing",
            "curious", "surprised", "affectionate", "sleepy", "sad",
            "concerned", "annoyed", "error"
    };
    private static final float BASE_W = 1080f;
    private static final float BASE_H = 720f;
    private static final float LEFT_CENTER_X = 294.5f;
    private static final float RIGHT_CENTER_X = BASE_W - LEFT_CENTER_X;
    private static final float EYE_CENTER_Y = 391f;
    // Production renderer: the accepted corgi shell. The M4 branch stays in
    // source only as an archived comparison renderer.
    private static final boolean M4_HAZEL_PREVIEW = false;
    // Reversible visual preview: keep the proven corgi expression rig, but
    // expose it through the two original physical eye envelopes rather than
    // drawing the complete head. Switching this flag back preserves v4.5's
    // full-face renderer without changing the semantic or animation stack.
    private static final boolean CIRCULAR_CORGI_EYES_PREVIEW = false;
    private static final long DIRECT_PREVIEW_TTL_MS = 15_000L;
    private static final long EVENT_TTL_MS = 60_000L;
    // More columns keep the generated fur ridge continuous while it bends;
    // ten wide strips were visible as discrete steps during large emotions.
    private static final int BROW_MESH_COLUMNS = 24;
    private static final int BROW_MESH_ROWS = 1;

    private final Paint globePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pupilPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint reflectionPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bitmapPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint lidEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint wetEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint browTexturePaint = new Paint(
            Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint m4SocketPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint m4LidPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint m4LidEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path eyePath = new Path();
    private final Path upperCoverPath = new Path();
    private final Path lowerCoverPath = new Path();
    private final Path upperBoundaryPath = new Path();
    private final Path lowerBoundaryPath = new Path();
    private final Path reflectionPath = new Path();
    private final Path m4EyePath = new Path();
    private final Path m4LidPath = new Path();
    private final float[] browMesh = new float[
            (BROW_MESH_COLUMNS + 1) * (BROW_MESH_ROWS + 1) * 2];
    private final float[] browMeshT = new float[BROW_MESH_COLUMNS + 1];
    private final float[] browMeshArc = new float[BROW_MESH_COLUMNS + 1];
    private final float[] browMeshTaper = new float[BROW_MESH_COLUMNS + 1];
    private final RectF assetRect = new RectF(0f, 0f, BASE_W, BASE_H);
    // A photo the robot chose to show takes the face's own visible frame for
    // a few seconds; the face returns afterwards (owner 06.10.2026).
    private final RectF photoRect = new RectF();
    private final Paint photoPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private Bitmap photo;
    private long photoStarted;
    private long photoUntil;
    private final RectF irisRect = new RectF();
    private final RectF m4ScleraRect = new RectF();
    private final Random random = new Random();

    private final Bitmap irisTexture;
    private final Bitmap furPlate;
    private final Bitmap closedFur;
    private final Bitmap browLeftTexture;
    private final Bitmap browRightTexture;
    private final Bitmap m4HazelIris;
    private final Bitmap m4HazelSclera;
    private final ApprovedCorgiSpriteRig corgiRig;

    private Emotion emotion = Emotion.IDLE;
    private Emotion baselineEmotion = Emotion.IDLE;
    private float baselineIntensity = 1f;
    private Emotion interactionEmotion;
    private float interactionIntensity = 1f;
    private long interactionExpiresAt;
    private String interactionEventId = "";
    private Emotion eventEmotion;
    private float eventIntensity = 1f;
    private long eventExpiresAt;
    private String eventEventId = "";
    private Emotion reactionEmotion;
    private float reactionIntensity = 1f;
    private long reactionExpiresAt;
    private String reactionEventId = "";
    private Emotion directEmotion;
    private float directIntensity = 1f;
    private long directExpiresAt;
    private String directEventId = "";
    private Emotion activityEmotion = Emotion.IDLE;
    private Emotion affectEmotion = Emotion.IDLE;
    private float affectIntensity = 1f;
    private long expressionSequence;
    private long expressionStarted = SystemClock.uptimeMillis();
    private long affectStarted = expressionStarted;
    private long transitionStarted = SystemClock.uptimeMillis();
    private float fromClosureLeft = 0.04f;
    private float fromClosureRight = 0.04f;
    private float fromPupil = 0.48f;
    private float fromIrisScale = 1f;
    private float displayedClosureLeft = fromClosureLeft;
    private float displayedClosureRight = fromClosureRight;
    private float displayedPupil = fromPupil;
    private float displayedIrisScale = fromIrisScale;
    // outerY, innerY, arch, strength for left followed by right. Keeping the
    // actually displayed values lets a new pose start without a visual pop.
    private final float[] fromBrowPose = {
            270f, 260f, 15f, 0.58f,
            270f, 260f, 15f, 0.58f
    };
    private final float[] displayedBrowPose = fromBrowPose.clone();

    private long nextBlink = transitionStarted + 4300;
    private long blinkStarted = -1;
    private int blinkCloseMs;
    private int blinkHoldMs;
    private int blinkOpenMs;
    private float blinkDepth;
    private long queuedBlinkAt = -1;
    private long queuedSecondBlinkAt = -1;

    private long targetGazeUntil;
    private float gazeX;
    private float gazeY;
    private float gazeFromX;
    private float gazeFromY;
    private float gazeTargetX;
    private float gazeTargetY;
    private long gazeStarted = transitionStarted;
    private int gazeDurationMs = 220;
    private long nextGaze = transitionStarted + 1200;
    // Ambient gaze is a shared binocular attention signal, never two random
    // eye positions.  The ordered targets give the face readable left, right,
    // up and down glances without the mechanical ping-pong of pure RNG.
    private static final float[][] AMBIENT_GAZE_TARGETS = {
            {-0.48f, -0.06f},
            { 0.42f,  0.10f},
            { 0.12f, -0.28f},
            {-0.35f,  0.18f},
            { 0.47f, -0.15f},
            {-0.10f,  0.26f},
            {-0.44f,  0.08f},
            { 0.31f,  0.22f},
    };
    private Emotion gazeActivity = Emotion.IDLE;
    private Emotion gazeAffect = Emotion.IDLE;
    private long gazeDecisionSequence;
    private int ambientGazeIndex;
    private boolean idleReturnToContactNext;
    // Purely visual idle life. It never changes the semantic/API state that
    // Hermes and the rest of RobotBrain observe.
    private boolean ambientIdleEligible;
    private Emotion ambientVisualAffect = Emotion.IDLE;
    private float ambientVisualBlend;
    private long ambientVisualStarted = -1L;
    private int ambientVisualDurationMs;
    private long nextAmbientVisual;
    private int lastAmbientVisualIndex = -1;
    private long ambientVisualExitStarted = -1L;
    private float ambientVisualExitFromBlend;
    private boolean staticWakeScheduled;
    private long staticWakeDeadline;
    private final Runnable staticWakeRunnable = new Runnable() {
        @Override public void run() {
            staticWakeScheduled = false;
            staticWakeDeadline = 0L;
            invalidate();
        }
    };

    private float downX;
    private float downY;
    private boolean cameraActive;
    private boolean microphoneActive;
    private boolean networkActive;
    private volatile float speechTarget;
    private volatile boolean speechActive;
    private volatile long speechUpdated;
    private float displayedSpeech;

    void setSpeechLevel(float level, boolean active) {
        if (active && !awake) post(() -> setDisplayAwake(true));
        speechTarget = Float.isFinite(level) ? clamp01(level) : 0f;
        speechActive = active;
        speechUpdated = SystemClock.uptimeMillis();
        postInvalidateOnAnimation();
    }

    public FaceView(Context context) {
        super(context);
        setBackgroundColor(Color.BLACK);
        // The window is already hardware accelerated. Forcing this full-screen
        // View into a separate hardware layer caused an extra offscreen texture
        // upload on every animated frame on the Snapdragon 845.

        globePaint.setStyle(Paint.Style.FILL);
        globePaint.setColor(Color.rgb(30, 17, 11));
        pupilPaint.setStyle(Paint.Style.FILL);
        pupilPaint.setColor(Color.rgb(5, 4, 3));
        reflectionPaint.setStyle(Paint.Style.FILL);
        lidEdgePaint.setStyle(Paint.Style.STROKE);
        lidEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        lidEdgePaint.setStrokeJoin(Paint.Join.ROUND);
        lidEdgePaint.setColor(Color.rgb(25, 14, 10));
        lidEdgePaint.setStrokeWidth(3.5f);
        wetEdgePaint.setStyle(Paint.Style.STROKE);
        wetEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        wetEdgePaint.setStrokeJoin(Paint.Join.ROUND);
        wetEdgePaint.setColor(Color.rgb(220, 205, 188));
        wetEdgePaint.setStrokeWidth(1.15f);
        wetEdgePaint.setAlpha(105);
        browTexturePaint.setAlpha(255);
        m4SocketPaint.setStyle(Paint.Style.STROKE);
        m4SocketPaint.setStrokeCap(Paint.Cap.ROUND);
        m4SocketPaint.setStrokeJoin(Paint.Join.ROUND);
        m4SocketPaint.setStrokeWidth(24f);
        m4SocketPaint.setColor(Color.rgb(46, 9, 8));
        m4LidPaint.setStyle(Paint.Style.FILL);
        // Matches the documented backColor of the official Hazel preset.
        m4LidPaint.setColor(Color.rgb(140, 40, 20));
        m4LidEdgePaint.setStyle(Paint.Style.STROKE);
        m4LidEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        m4LidEdgePaint.setStrokeWidth(5f);
        m4LidEdgePaint.setColor(Color.rgb(238, 118, 67));

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        // The previous eye-only shell stays in source as a comparison renderer,
        // but its large textures are not decoded in production. The complete
        // corgi renderer owns the two matched full-face plates and its local rig.
        irisTexture = null;
        furPlate = null;
        closedFur = null;
        browLeftTexture = null;
        browRightTexture = null;
        corgiRig = new ApprovedCorgiSpriteRig(
                context, CIRCULAR_CORGI_EYES_PREVIEW);
        // MIT-licensed Adafruit M4 Eyes Hazel textures. Attribution is bundled
        // in assets/licenses/adafruit-m4-eyes.txt. Do not decode the inactive
        // comparison renderer into production memory or upload it to the GPU.
        if (M4_HAZEL_PREVIEW) {
            m4HazelIris = BitmapFactory.decodeResource(
                    getResources(), R.drawable.m4_hazel_iris, options);
            m4HazelSclera = BitmapFactory.decodeResource(
                    getResources(), R.drawable.m4_hazel_sclera, options);
        } else {
            m4HazelIris = null;
            m4HazelSclera = null;
        }
        for (int column = 0; column <= BROW_MESH_COLUMNS; column++) {
            float t = column / (float) BROW_MESH_COLUMNS;
            float arc = (float)Math.sin(Math.PI * t);
            browMeshT[column] = t;
            browMeshArc[column] = arc;
            browMeshTaper[column] = 0.78f + 0.22f * arc;
        }
        updateContentDescriptionLocked();
    }

    public static boolean isKnownEmotion(String value) {
        if (value == null) return false;
        for (String name : NAMES) {
            if (name.equalsIgnoreCase(value.trim())) return true;
        }
        return false;
    }

    public static String allowedAsJson() {
        StringBuilder out = new StringBuilder("[");
        for (int index = 0; index < NAMES.length; index++) {
            if (index > 0) out.append(',');
            out.append('"').append(NAMES[index]).append('"');
        }
        return out.append(']').toString();
    }

    public synchronized String getEmotionName() {
        return emotion.name().toLowerCase(Locale.US);
    }

    public synchronized String stateAsJson() {
        long now = SystemClock.uptimeMillis();
        return "{\"state\":\"" + emotion.name().toLowerCase(Locale.US) +
                "\",\"activity\":\"" + activityEmotion.name().toLowerCase(Locale.US) +
                "\",\"affect\":\"" + affectEmotion.name().toLowerCase(Locale.US) +
                "\",\"body\":\"" + baselineEmotion.name().toLowerCase(Locale.US) +
                "\",\"voice\":" + emotionJson(interactionEmotion) +
                ",\"voice_event_id\":\"" + interactionEventId + "\"" +
                ",\"event\":" + emotionJson(eventEmotion) +
                ",\"event_event_id\":\"" + eventEventId + "\"" +
                ",\"reaction\":" + emotionJson(reactionEmotion) +
                ",\"reaction_event_id\":\"" + reactionEventId + "\"" +
                ",\"direct\":" + emotionJson(directEmotion) +
                ",\"direct_event_id\":\"" + directEventId + "\"" +
                ",\"affect_intensity\":" + formatFloat(affectIntensity) +
                ",\"gaze_x\":" + formatFloat(gazeX) +
                ",\"gaze_y\":" + formatFloat(gazeY) +
                ",\"target_gaze_active\":" + (now < targetGazeUntil) +
                ",\"sequence\":" + expressionSequence +
                ",\"voice_ttl_ms\":" + remainingMs(interactionExpiresAt, now) +
                ",\"event_ttl_ms\":" + remainingMs(eventExpiresAt, now) +
                ",\"reaction_ttl_ms\":" + remainingMs(reactionExpiresAt, now) +
                ",\"speech_active\":" + (speechActive && now - speechUpdated < 250L) +
                ",\"mouth_open\":" + formatFloat(displayedSpeech) +
                ",\"direct_ttl_ms\":" + remainingMs(directExpiresAt, now) + "}";
    }

    private static String emotionJson(Emotion value) {
        return value == null ? "null" : "\"" +
                value.name().toLowerCase(Locale.US) + "\"";
    }

    private static long remainingMs(long deadline, long now) {
        return deadline <= now ? 0L : deadline - now;
    }

    private static String formatFloat(float value) {
        return String.format(Locale.US, "%.3f", value);
    }

    public synchronized void setEmotion(String name) {
        setEmotion(name, "direct", "", 1f);
    }

    public synchronized void setEmotion(String name, String source) {
        setEmotion(name, source, "", 1f);
    }

    public synchronized void setEmotion(String name, String source, String eventId) {
        setEmotion(name, source, eventId, 1f);
    }

    public synchronized void setEmotion(String name, String source, String eventId,
                                        float intensity) {
        if (!isKnownEmotion(name)) return;
        Emotion next = Emotion.valueOf(name.trim().toUpperCase(Locale.US));
        String normalizedSource = source == null ? "direct" :
                source.trim().toLowerCase(Locale.US);
        String normalizedEventId = eventId == null ? "" : eventId.trim();
        float normalizedIntensity = clamp01(intensity);
        long now = SystemClock.uptimeMillis();
        boolean duplicateDelivery = false;
        if ("body".equals(normalizedSource)) {
            duplicateDelivery = next == baselineEmotion;
            baselineEmotion = next;
            baselineIntensity = normalizedIntensity;
        } else if ("voice".equals(normalizedSource)) {
            duplicateDelivery = next == interactionEmotion &&
                    !normalizedEventId.isEmpty() &&
                    normalizedEventId.equals(interactionEventId);
            if (next == Emotion.IDLE) {
                interactionEmotion = null;
                interactionIntensity = 1f;
                interactionExpiresAt = 0L;
                interactionEventId = "";
            } else {
                interactionEmotion = next;
                interactionIntensity = normalizedIntensity;
                interactionExpiresAt = now + interactionTtlMs(next);
                interactionEventId = normalizedEventId;
            }
        } else if ("event".equals(normalizedSource)) {
            duplicateDelivery = next == eventEmotion &&
                    !normalizedEventId.isEmpty() &&
                    normalizedEventId.equals(eventEventId);
            if (next == Emotion.IDLE) {
                eventEmotion = null;
                eventIntensity = 1f;
                eventExpiresAt = 0L;
                eventEventId = "";
            } else {
                eventEmotion = next;
                eventIntensity = normalizedIntensity;
                eventExpiresAt = now + EVENT_TTL_MS;
                eventEventId = normalizedEventId;
            }
        } else if ("reaction".equals(normalizedSource)) {
            duplicateDelivery = next == reactionEmotion &&
                    !normalizedEventId.isEmpty() &&
                    normalizedEventId.equals(reactionEventId);
            if (next == Emotion.IDLE) {
                reactionEmotion = null;
                reactionIntensity = 1f;
                reactionExpiresAt = 0L;
                reactionEventId = "";
            } else {
                reactionEmotion = next;
                reactionIntensity = normalizedIntensity;
                reactionExpiresAt = now + reactionTtlMs(next);
                reactionEventId = normalizedEventId;
            }
        } else {
            duplicateDelivery = next == directEmotion &&
                    !normalizedEventId.isEmpty() &&
                    normalizedEventId.equals(directEventId);
            // Direct calls and screen taps are bounded previews. They must not
            // overwrite a live body alarm that Psyche will not resend until
            // its source state changes.
            if (next == Emotion.IDLE) {
                directEmotion = null;
                directIntensity = 1f;
                directExpiresAt = 0L;
                directEventId = "";
            } else {
                directEmotion = next;
                directIntensity = normalizedIntensity;
                directExpiresAt = now + DIRECT_PREVIEW_TTL_MS;
                directEventId = normalizedEventId;
            }
        }
        // A heartbeat repeats the same event id to refresh delivery and TTL;
        // it must not restart the gesture, blink queue or gaze. A genuinely
        // new communicative act carries a fresh id and is replayed once.
        boolean replayEntry = !"body".equals(normalizedSource) &&
                !duplicateDelivery;
        boolean replayAffect = "reaction".equals(normalizedSource) ||
                "event".equals(normalizedSource) || "direct".equals(normalizedSource);
        applyResolvedExpressionLocked(now, replayEntry,
                replayAffect && !duplicateDelivery);
    }

    private static long interactionTtlMs(Emotion value) {
        switch (value) {
            case LISTENING: return 25_000L;
            case THINKING: return 60_000L;
            case SPEAKING: return 120_000L;
            case ERROR: return 15_000L;
            default: return 45_000L;
        }
    }

    private static long reactionTtlMs(Emotion value) {
        switch (value) {
            case SURPRISED: return 4_500L;
            case LAUGHING: return 6_000L;
            case CURIOUS: return 8_000L;
            case HAPPY: return 9_000L;
            case ANNOYED:
            case ERROR: return 7_000L;
            case SAD: return 12_000L;
            case AFFECTIONATE:
            case CONCERNED: return 14_000L;
            default: return 10_000L;
        }
    }

    private static boolean nearlyEqual(float left, float right) {
        return Math.abs(left - right) < 0.005f;
    }

    private static boolean isActivity(Emotion value) {
        return value == Emotion.LISTENING || value == Emotion.THINKING ||
                value == Emotion.SPEAKING;
    }

    private Emotion resolveActivityLocked() {
        if (isActivity(interactionEmotion)) return interactionEmotion;
        if (isActivity(directEmotion)) return directEmotion;
        if (isActivity(eventEmotion)) return eventEmotion;
        return Emotion.IDLE;
    }

    private Emotion resolveAffectLocked() {
        // Physical distress is an affect overlay, not a replacement for the
        // listening/thinking/speaking channel. The activity remains legible
        // while the eyes still communicate the body truth.
        if (baselineEmotion == Emotion.ERROR || baselineEmotion == Emotion.CONCERNED) {
            return baselineEmotion;
        }
        if (interactionEmotion == Emotion.ERROR) return interactionEmotion;
        // A newly proven mission outcome outranks the conversational pose that
        // happened to precede it; otherwise "arrived" could remain hidden
        // behind a fourteen-second social reaction.
        if (eventEmotion != null && !isActivity(eventEmotion)) return eventEmotion;
        if (reactionEmotion != null && !isActivity(reactionEmotion)) return reactionEmotion;
        if (directEmotion != null && !isActivity(directEmotion)) return directEmotion;
        return isActivity(baselineEmotion) ? Emotion.IDLE : baselineEmotion;
    }

    private float resolveAffectIntensityLocked() {
        if (baselineEmotion == Emotion.ERROR || baselineEmotion == Emotion.CONCERNED) {
            return baselineIntensity;
        }
        if (interactionEmotion == Emotion.ERROR) return interactionIntensity;
        if (eventEmotion != null && !isActivity(eventEmotion)) return eventIntensity;
        if (reactionEmotion != null && !isActivity(reactionEmotion)) {
            return reactionIntensity;
        }
        if (directEmotion != null && !isActivity(directEmotion)) return directIntensity;
        return isActivity(baselineEmotion) ? 1f : baselineIntensity;
    }

    private void applyResolvedExpressionLocked(long now, boolean replayEntry,
                                               boolean replayAffect) {
        Emotion nextActivity = resolveActivityLocked();
        Emotion nextAffect = resolveAffectLocked();
        float nextAffectIntensity = resolveAffectIntensityLocked();
        if (!replayEntry && nextActivity == activityEmotion && nextAffect == affectEmotion &&
                nearlyEqual(nextAffectIntensity, affectIntensity)) {
            return;
        }

        // Start at the actually displayed pose so rapid state changes do not
        // snap through a stale enum preset.
        fromClosureLeft = displayedClosureLeft;
        fromClosureRight = displayedClosureRight;
        fromPupil = displayedPupil;
        fromIrisScale = displayedIrisScale;
        System.arraycopy(displayedBrowPose, 0, fromBrowPose, 0,
                displayedBrowPose.length);
        Emotion previousActivity = activityEmotion;
        Emotion previousAffect = affectEmotion;
        activityEmotion = nextActivity;
        affectEmotion = nextAffect;
        affectIntensity = nextAffectIntensity;
        emotion = nextActivity != Emotion.IDLE ? nextActivity : nextAffect;
        transitionStarted = now;
        expressionStarted = now;
        if (nextAffect != previousAffect || replayAffect) affectStarted = now;
        expressionSequence++;
        nextGaze = now;

        boolean affectEntry = nextAffect != previousAffect || replayAffect;
        boolean listeningEntry = nextActivity == Emotion.LISTENING &&
                nextActivity != previousActivity;
        if (affectEntry) {
            queuedBlinkAt = -1;
            queuedSecondBlinkAt = -1;
        }
        if (listeningEntry) {
            // The wake acknowledgement is an open-eyed mutual-gaze beat. Drop
            // any ambient blink already in flight so it cannot hide "I heard
            // you" during the first readable key pose.
            blinkStarted = -1;
            queuedBlinkAt = -1;
            queuedSecondBlinkAt = -1;
        }
        if (!listeningEntry && affectEntry && (nextAffect == Emotion.HAPPY ||
                nextAffect == Emotion.AFFECTIONATE)) {
            // First establish the smile/warm gaze; only then add the friendly
            // double blink as a secondary accent. An entry blink used to hide
            // the expression before its 320 ms transition had even settled.
            queuedBlinkAt = now + 900;
            queuedSecondBlinkAt = now + 1_560;
        } else if (!listeningEntry && affectEntry && nextAffect == Emotion.SAD) {
            queuedBlinkAt = now + 1_050;
        }
        scheduleNextBlink(now, nextActivity, nextAffect);
        updateContentDescriptionLocked();
        cancelStaticWake();
        invalidate();
    }

    /** Keep privacy state independent from the expressive face state. */
    public synchronized void setPrivacyActivity(boolean camera, boolean microphone,
                                                boolean network) {
        if (cameraActive == camera && microphoneActive == microphone &&
                networkActive == network) return;
        cameraActive = camera;
        microphoneActive = microphone;
        networkActive = network;
        updateContentDescriptionLocked();
        invalidate();
    }

    private void updateContentDescriptionLocked() {
        setContentDescription("Robot face: " + emotion.name().toLowerCase(Locale.US) +
                "; activity " + activityEmotion.name().toLowerCase(Locale.US) +
                "; affect " + affectEmotion.name().toLowerCase(Locale.US) +
                "; camera " + (cameraActive ? "active" : "inactive") +
                "; microphone " + (microphoneActive ? "active" : "inactive") +
                "; network " + (networkActive ? "active" : "inactive"));
    }

    // ---------------------------------------------------------------- dark idle screen
    private static final long SLEEP_FADE_MS = 600L, WAKE_FADE_MS = 250L, EYES_OPEN_MS = 300L;  // 19.10: woken by «Братан» — at once
    private volatile boolean awake = true;
    private volatile long displayChanged = -100_000L;
    private final Paint darkPaint = new Paint();

    /** Dark when idle (AMOLED black = no light, no drawing); waking fades the face in and opens the eyes. */
    public synchronized void setDisplayAwake(boolean value) {
        if (value == awake) return;
        long now = SystemClock.uptimeMillis();
        // Reverse a fade in progress from where it is, not from the start.
        long elapsed = now - displayChanged;
        long fade = awake ? SLEEP_FADE_MS : WAKE_FADE_MS;
        long carry = elapsed < fade ? fade - elapsed : 0L;
        awake = value;
        displayChanged = now - carry;
        cancelStaticWake();
        postInvalidateOnAnimation();
    }

    public boolean displayAwake() { return awake; }

    /** 0 = fully visible, 1 = fully dark. */
    private float darkness(long now) {
        long elapsed = now - displayChanged;
        if (awake) return 1f - clamp01(elapsed / (float) WAKE_FADE_MS);
        return clamp01(elapsed / (float) SLEEP_FADE_MS);
    }

    /** Extra eyelid closure while waking: shut, then opening just after the fade. */
    private float wakeClosure(long now) {
        if (!awake) return clamp01(darkness(now) * 1.5f);   // the eyes close as the face fades out
        float t = (now - displayChanged - WAKE_FADE_MS * 0.5f) / EYES_OPEN_MS;
        return 1f - smoothStep(clamp01(t));
    }

    public synchronized void showPhoto(Bitmap bitmap, long durationMs) {
        photo = bitmap;
        photoStarted = SystemClock.uptimeMillis();
        photoUntil = photoStarted + Math.max(1000L, Math.min(30_000L, durationMs));
        cancelStaticWake();
        postInvalidateOnAnimation();
    }

    public synchronized void clearPhoto() {
        photo = null;
        photoUntil = 0L;
        postInvalidateOnAnimation();
    }

    public synchronized String photoStateJson() {
        long left = Math.max(0L, photoUntil - SystemClock.uptimeMillis());
        return "{\"showing\":" + (photo != null && left > 0L) + ",\"remaining_ms\":" + left + "}";
    }

    /** Draws the photo inside the visible face frame; false when none is due. */
    private boolean drawPhoto(Canvas canvas, long now) {
        Bitmap shown;
        long started, until;
        synchronized (this) {
            if (photo != null && now >= photoUntil) photo = null;
            shown = photo;
            started = photoStarted;
            until = photoUntil;
        }
        if (shown == null) return false;
        float activeHeight = Math.min(getHeight() / 3f, getWidth() * BASE_H / BASE_W);
        float fade = Math.min(1f, Math.min(now - started, until - now) / 300f);
        canvas.save();
        canvas.clipRect(0f, 0f, getWidth(), activeHeight);
        canvas.drawColor(Color.BLACK);
        float scale = Math.min(getWidth() / (float) shown.getWidth(), activeHeight / shown.getHeight());
        float w = shown.getWidth() * scale, h = shown.getHeight() * scale;
        photoRect.set((getWidth() - w) / 2f, (activeHeight - h) / 2f,
                (getWidth() + w) / 2f, (activeHeight + h) / 2f);
        photoPaint.setAlpha(Math.round(255f * Math.max(0f, fade)));
        canvas.drawBitmap(shown, null, photoRect, photoPaint);
        canvas.restore();
        if (fade < 1f) postInvalidateOnAnimation();
        else postInvalidateDelayed(Math.max(16L, Math.min(until - now - 300L, 1000L)));
        return true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        long now = SystemClock.uptimeMillis();
        float dark = darkness(now);
        if (!awake && dark >= 1f) {
            canvas.drawColor(Color.BLACK);       // asleep: nothing to draw, nothing scheduled
            return;
        }
        if (drawPhoto(canvas, now)) return;
        Emotion activity;
        Emotion affect;
        synchronized (this) {
            boolean expired = false;
            if (interactionEmotion != null && now >= interactionExpiresAt) {
                interactionEmotion = null;
                interactionIntensity = 1f;
                interactionExpiresAt = 0L;
                interactionEventId = "";
                expired = true;
            }
            if (eventEmotion != null && now >= eventExpiresAt) {
                eventEmotion = null;
                eventIntensity = 1f;
                eventExpiresAt = 0L;
                eventEventId = "";
                expired = true;
            }
            if (reactionEmotion != null && now >= reactionExpiresAt) {
                reactionEmotion = null;
                reactionIntensity = 1f;
                reactionExpiresAt = 0L;
                reactionEventId = "";
                expired = true;
            }
            if (directEmotion != null && now >= directExpiresAt) {
                directEmotion = null;
                directIntensity = 1f;
                directExpiresAt = 0L;
                directEventId = "";
                expired = true;
            }
            if (expired) applyResolvedExpressionLocked(now, false, false);
            activity = activityEmotion;
            affect = affectEmotion;
        }
        boolean actualSpeech = speechActive && now - speechUpdated < 250L;
        float jawTarget = actualSpeech ? speechTarget : 0f;
        displayedSpeech = lerp(displayedSpeech, jawTarget,
                jawTarget > displayedSpeech ? 0.55f : 0.25f);
        if (displayedSpeech < 0.002f) displayedSpeech = 0f;
        if (actualSpeech) activity = Emotion.SPEAKING;

        updateBlink(now, activity, affect);
        updateGaze(now, activity, affect);
        updateAmbientVisual(now, activity, affect);
        float elapsed = now - transitionStarted;
        // Social-robot expressions have to read from across the room. Keep the
        // entry decisive, but give every channel enough display frames to make
        // the morph continuous on the handset's 60 Hz panel.
        float transitionMs;
        if (affect == Emotion.SURPRISED || activity == Emotion.LISTENING) {
            transitionMs = 270f;
        } else if (affect == Emotion.SAD || affect == Emotion.CONCERNED ||
                affect == Emotion.SLEEPY) {
            transitionMs = 480f;
        } else if (affect == Emotion.HAPPY || affect == Emotion.LAUGHING ||
                affect == Emotion.AFFECTIONATE) {
            transitionMs = 380f;
        } else {
            transitionMs = 340f;
        }
        // The stable channels settle monotonically. Only the diagnostic fur
        // ridge gets a small cartoon overshoot; globally overshooting pupil,
        // iris and closure made unrelated cues flare together.
        float transition = smoothStep(elapsed / transitionMs);
        float browTransition = cartoonPoseProgress(elapsed / transitionMs);
        float blink = blinkAmount(now);

        // Eyelids are a binocular unit. Emotional winks and alternating error
        // glitches made the character look neurologically incoherent, so the
        // complete resting pose, transition and blink phase are shared.
        float fromClosure = 0.5f * (fromClosureLeft + fromClosureRight);
        float sharedClosure = clamp01(lerp(fromClosure,
                targetClosure(activity, affect, now), transition));
        sharedClosure = sharedClosure + (1f - sharedClosure) * blink;
        sharedClosure = Math.max(sharedClosure, wakeClosure(now));
        float closureLeft = sharedClosure;
        float closureRight = sharedClosure;
        float pupil = clamp(lerp(fromPupil,
                targetPupil(activity, affect, now), transition), 0.22f, 0.76f);
        float irisScale = clamp(lerp(fromIrisScale,
                targetIrisScale(activity, affect, now), transition), 0.94f, 1.08f);

        synchronized (this) {
            displayedClosureLeft = closureLeft;
            displayedClosureRight = closureRight;
            displayedPupil = pupil;
            displayedIrisScale = irisScale;
        }

        float activeHeight = Math.min(getHeight() / 3f, getWidth() * BASE_H / BASE_W);
        float scale = Math.min(getWidth() / BASE_W, activeHeight / BASE_H);
        float offsetX = (getWidth() - BASE_W * scale) * 0.5f;

        canvas.save();
        // Do not draw anything below activeHeight: the lower OLED region is
        // intentionally outside the face surface on this handset.
        canvas.clipRect(0f, 0f, getWidth(), activeHeight);
        canvas.translate(offsetX, 0f);
        canvas.scale(scale, scale);
        if (M4_HAZEL_PREVIEW) {
            drawM4HazelEye(canvas, true, closureLeft, irisScale, pupil,
                    activity, affect, now);
            drawM4HazelEye(canvas, false, closureRight, irisScale, pupil,
                    activity, affect, now);
        } else {
            corgiRig.draw(canvas,
                    NAMES[activity.ordinal()],
                    NAMES[affect.ordinal()],
                    affectIntensity, blink, gazeX, gazeY, irisScale, pupil,
                    sharedClosure,
                    NAMES[ambientVisualAffect.ordinal()],
                    ambientVisualBlend,
                    now, expressionStarted, affectStarted, expressionSequence,
                    actualSpeech || activity == Emotion.SPEAKING || displayedSpeech > 0f
                            ? displayedSpeech : -1f);
        }
        canvas.restore();
        if (dark > 0f) {
            darkPaint.setColor(Color.BLACK);
            darkPaint.setAlpha(Math.round(255f * dark));
            canvas.drawRect(0f, 0f, getWidth(), getHeight(), darkPaint);
        }

        boolean transitionMoving = elapsed < transitionMs || now - displayChanged < WAKE_FADE_MS + EYES_OPEN_MS + 100L
                || (!awake && dark < 1f);
        // A queued blink is a future deadline, not ongoing animation. Treating
        // the 900/1560 ms waits as movement needlessly rendered at 60 fps.
        boolean blinkMoving = blinkStarted >= 0;
        boolean gazeMoving = now - gazeStarted < gazeDurationMs;
        boolean ambientVisualMoving = isAmbientVisualMoving(now);
        long actionElapsed = Math.max(0L, now - expressionStarted);
        long affectElapsed = Math.max(0L, now - affectStarted);
        boolean gestureMoving = M4_HAZEL_PREVIEW &&
                ((activity == Emotion.LISTENING && actionElapsed < 720L) ||
                activity == Emotion.SPEAKING ||
                (affect == Emotion.LAUGHING && affectElapsed < 3_200L) ||
                (affect == Emotion.ERROR && affectElapsed < 900L));
        // postInvalidateDelayed(16) is not phase-locked to display refresh and
        // can alternate between one and two vsync intervals under load. While
        // anything is moving, request the next display frame directly. Static
        // holds stay event-light, so smooth acting is not a permanent 60 fps
        // battery tax.
        if (transitionMoving || blinkMoving || gazeMoving || gestureMoving ||
                ambientVisualMoving || actualSpeech || displayedSpeech > 0f) {
            cancelStaticWake();
            postInvalidateOnAnimation();
        } else {
            scheduleStaticWake(now);
        }
    }

    /**
     * Android adaptation of Adafruit's M4 Eyes Hazel visual preset. The
     * upstream texture maps are used with a clean local Canvas renderer; the
     * RobotBrain state, timing and capability API remain unchanged.
     */
    private void drawM4HazelEye(Canvas canvas, boolean left, float closure,
                                float irisScale, float pupilScale,
                                Emotion activity, Emotion affect, long now) {
        buildM4Eye(left);
        canvas.drawPath(m4EyePath, m4SocketPaint);
        int clip = canvas.save();
        canvas.clipPath(m4EyePath);

        float centerX = left ? 280f : 800f;
        float centerY = 362f;
        float actionElapsed = Math.max(0L, now - expressionStarted);
        float affectElapsed = Math.max(0L, now - affectStarted);
        float gestureX = 0f;
        float gestureY = 0f;
        if (affect == Emotion.IDLE && activity == Emotion.LISTENING &&
                actionElapsed < 720L) {
            gestureY -= 17f * (1f - smoothStep(actionElapsed / 720f));
        } else if (activity == Emotion.SPEAKING) {
            gestureY += 7f * (float)Math.sin(now / 105.0);
        }
        if (affect == Emotion.LAUGHING && affectElapsed < 3200L) {
            gestureY += 12f * (float)Math.sin(affectElapsed / 72.0);
        } else if (affect == Emotion.ERROR) {
            gestureX += 8f * (float)Math.sin(now / 43.0);
            gestureY += 5f * (float)Math.sin(now / 57.0);
        }

        // The vessel texture is wider than one aperture, matching the
        // upstream spherical-map intent while still rendering cheaply.
        float scleraShiftX = gazeX * 24f;
        float scleraShiftY = gazeY * 10f;
        m4ScleraRect.set(centerX - 255f + scleraShiftX,
                centerY - 142f + scleraShiftY,
                centerX + 255f + scleraShiftX,
                centerY + 142f + scleraShiftY);
        if (m4HazelSclera != null) {
            canvas.drawBitmap(m4HazelSclera, null, m4ScleraRect, bitmapPaint);
        } else {
            globePaint.setColor(Color.rgb(235, 218, 202));
            canvas.drawPath(m4EyePath, globePaint);
        }

        float irisX = centerX + gazeX * 82f + gestureX;
        float irisY = centerY + gazeY * 54f + gestureY;
        float radius = 105f * Math.max(0.92f, Math.min(1.12f, irisScale));
        irisRect.set(irisX - radius, irisY - radius,
                irisX + radius, irisY + radius);
        if (m4HazelIris != null) {
            canvas.drawBitmap(m4HazelIris, null, irisRect, bitmapPaint);
        }
        float humanPupil = 0.19f + 0.40f * clamp01(
                (pupilScale - 0.22f) / 0.56f);
        canvas.drawCircle(irisX, irisY, radius * humanPupil, pupilPaint);
        reflectionPaint.setColor(Color.WHITE);
        reflectionPaint.setAlpha(218);
        canvas.drawCircle(irisX - radius * 0.30f,
                irisY - radius * 0.34f, radius * 0.13f, reflectionPaint);
        reflectionPaint.setAlpha(118);
        canvas.drawCircle(irisX + radius * 0.11f,
                irisY - radius * 0.12f, radius * 0.045f, reflectionPaint);
        reflectionPaint.setAlpha(255);

        drawM4Lids(canvas, left, closure, activity, affect);
        canvas.restoreToCount(clip);
    }

    private void buildM4Eye(boolean left) {
        float outer = x(left, 70f);
        float inner = x(left, 490f);
        m4EyePath.rewind();
        m4EyePath.moveTo(outer, 360f);
        m4EyePath.cubicTo(x(left, 132f), 228f,
                x(left, 376f), 218f, inner, 354f);
        m4EyePath.cubicTo(x(left, 456f), 472f,
                x(left, 176f), 505f, outer, 360f);
        m4EyePath.close();
    }

    private void drawM4Lids(Canvas canvas, boolean left, float closure,
                            Emotion activity, Emotion affect) {
        float upper = clamp01(closure);
        float lower = clamp01(closure);
        if (affect == Emotion.HAPPY || affect == Emotion.AFFECTIONATE) {
            upper = closure * 0.14f;
            lower = closure * 1.95f;
        } else if (affect == Emotion.LAUGHING) {
            upper = closure * 0.48f;
            lower = closure * 1.55f;
        }
        upper = clamp01(upper);
        lower = clamp01(lower);
        float fullMix = smoothStep(clamp01((Math.max(upper, lower) - 0.72f) / 0.28f));
        float upperSlope = targetUpperLidSlope(activity, affect, left) *
                (1f - fullMix) * 0.58f;
        float lowerSlope = targetLowerLidSlope(activity, affect, left) *
                (1f - fullMix) * 0.42f;
        float centerX = left ? 280f : 800f;
        float outer = x(left, 70f);
        float inner = x(left, 490f);

        float upperCenterY = lerp(225f, 373f, upper);
        float upperOuterY = upperCenterY + (outer - centerX) * upperSlope;
        float upperInnerY = upperCenterY + (inner - centerX) * upperSlope;
        m4LidPath.rewind();
        m4LidPath.moveTo(outer, 190f);
        m4LidPath.lineTo(inner, 190f);
        m4LidPath.lineTo(inner, upperInnerY);
        m4LidPath.cubicTo(x(left, 392f), upperCenterY - 25f,
                x(left, 164f), upperCenterY - 18f, outer, upperOuterY);
        m4LidPath.close();
        canvas.drawPath(m4LidPath, m4LidPaint);
        m4LidPath.rewind();
        m4LidPath.moveTo(outer, upperOuterY);
        m4LidPath.cubicTo(x(left, 164f), upperCenterY - 18f,
                x(left, 392f), upperCenterY - 25f, inner, upperInnerY);
        canvas.drawPath(m4LidPath, m4LidEdgePaint);

        float lowerCenterY = lerp(505f, 373f, lower);
        float lowerOuterY = lowerCenterY + (outer - centerX) * lowerSlope;
        float lowerInnerY = lowerCenterY + (inner - centerX) * lowerSlope;
        m4LidPath.rewind();
        m4LidPath.moveTo(outer, lowerOuterY);
        m4LidPath.cubicTo(x(left, 176f), lowerCenterY + 18f,
                x(left, 404f), lowerCenterY + 12f, inner, lowerInnerY);
        m4LidPath.lineTo(inner, 535f);
        m4LidPath.lineTo(outer, 535f);
        m4LidPath.close();
        canvas.drawPath(m4LidPath, m4LidPaint);
        m4LidPath.rewind();
        m4LidPath.moveTo(outer, lowerOuterY);
        m4LidPath.cubicTo(x(left, 176f), lowerCenterY + 18f,
                x(left, 404f), lowerCenterY + 12f, inner, lowerInnerY);
        canvas.drawPath(m4LidPath, m4LidEdgePaint);
    }

    private void drawGlobe(Canvas canvas, boolean left, float irisScale,
                           float pupilScale, Emotion activity, Emotion affect,
                           long now) {
        buildGlobeClip(left);
        canvas.drawPath(eyePath, globePaint);
        int clip = canvas.save();
        canvas.clipPath(eyePath);

        float centerX = left ? LEFT_CENTER_X : RIGHT_CENTER_X;
        // A canine iris fills most of the visible aperture; keeping only small
        // dark canthi avoids the mannequin-like human almond-eye read.
        float radius = 96.5f * irisScale;
        // Normalized gaze is deliberately readable from across a room. The
        // former 29x18 range made listening and speaking shifts sub-pixel.
        float actionElapsed = Math.max(0L, now - expressionStarted);
        float affectElapsed = Math.max(0L, now - affectStarted);
        float gestureX = 0f;
        float gestureY = 0f;
        if (activity == Emotion.LISTENING && actionElapsed < 720L) {
            // A visible "I heard my name" pop: snap to the owner, rise, then
            // settle. This remains inside the dynamic aperture.
            float settle = smoothStep(actionElapsed / 720f);
            gestureY -= 18f * (1f - settle);
        } else if (affect == Emotion.IDLE && activity == Emotion.SPEAKING) {
            gestureY += 2.5f * speakingPulse(now);
        }
        if (affect == Emotion.LAUGHING && affectElapsed < 3200L) {
            gestureY += 11f * laughingPulse(now);
        } else if (affect == Emotion.ERROR) {
            float sharedError = errorPulse(now);
            gestureX += 5f * sharedError;
            gestureY += 3f * sharedError;
        }
        // One conjugate version signal moves both eyes. A tiny symmetric
        // convergence keeps near mutual gaze focused without looking crossed.
        float convergence = 2.5f + 2.0f * (1f - Math.min(1f, Math.abs(gazeX)));
        float irisX = centerX + gazeX * 57f + gestureX +
                (left ? convergence : -convergence);
        float irisY = EYE_CENTER_Y + gazeY * 39f + gestureY;
        irisRect.set(irisX - radius, irisY - radius, irisX + radius, irisY + radius);
        if (irisTexture != null) {
            canvas.drawBitmap(irisTexture, null, irisRect, bitmapPaint);
        } else {
            int oldColor = globePaint.getColor();
            globePaint.setColor(Color.rgb(150, 76, 24));
            canvas.drawCircle(irisX, irisY, radius, globePaint);
            globePaint.setColor(oldColor);
        }

        canvas.drawCircle(irisX, irisY, radius * pupilScale, pupilPaint);

        // Reflections stay attached to a virtual room light while the iris moves.
        reflectionPaint.setColor(Color.rgb(255, 246, 224));
        reflectionPaint.setAlpha(188);
        reflectionPath.rewind();
        reflectionPath.moveTo(centerX - 42f, EYE_CENTER_Y - 44f);
        reflectionPath.cubicTo(centerX - 35f, EYE_CENTER_Y - 45f,
                centerX - 22f, EYE_CENTER_Y - 42f,
                centerX - 14f, EYE_CENTER_Y - 40f);
        reflectionPath.cubicTo(centerX - 13f, EYE_CENTER_Y - 34f,
                centerX - 15f, EYE_CENTER_Y - 25f,
                centerX - 18f, EYE_CENTER_Y - 20f);
        reflectionPath.cubicTo(centerX - 25f, EYE_CENTER_Y - 20f,
                centerX - 34f, EYE_CENTER_Y - 22f,
                centerX - 38f, EYE_CENTER_Y - 24f);
        reflectionPath.cubicTo(centerX - 40f, EYE_CENTER_Y - 30f,
                centerX - 42f, EYE_CENTER_Y - 38f,
                centerX - 42f, EYE_CENTER_Y - 44f);
        reflectionPath.close();
        canvas.drawPath(reflectionPath, reflectionPaint);
        reflectionPaint.setAlpha(118);
        canvas.drawCircle(centerX - 9f, EYE_CENTER_Y - 15f, 4.2f, reflectionPaint);
        reflectionPaint.setAlpha(255);
        canvas.restoreToCount(clip);
    }

    /** Human-readable brow grammar rendered as a deforming ridge in the fur. */
    private void drawBrows(Canvas canvas, Emotion activity, Emotion affect,
                           long now, float transition) {
        drawBrow(canvas, true, activity, affect, now, transition);
        drawBrow(canvas, false, activity, affect, now, transition);
    }

    private void drawBrow(Canvas canvas, boolean left, Emotion activity,
                          Emotion affect, long now, float transition) {
        float outerY = 270f;
        float innerY = 260f;
        float arch = 15f;
        // The ridge is anatomy, not an optional graphic overlay. Even neutral
        // keeps a broad, textured mass so no transition can make it vanish.
        float strength = 0.58f;
        switch (affect) {
            case HAPPY:
                outerY = 252f; innerY = 247f; arch = 18f; strength = 0.90f; break;
            case LAUGHING:
                outerY = 244f; innerY = 240f; arch = 22f; strength = 1f; break;
            case CURIOUS:
                // Interest is open and bilateral. A permanently cocked single
                // brow read as scepticism and could disagree with gaze side.
                outerY = 250f; innerY = 224f; arch = 24f;
                strength = 0.94f; break;
            case SURPRISED:
                outerY = 222f; innerY = 214f; arch = 30f; strength = 1f; break;
            case AFFECTIONATE:
                outerY = 258f; innerY = 236f; arch = 18f; strength = 0.90f; break;
            case SLEEPY:
                outerY = 286f; innerY = 280f; arch = 7f; strength = 0.74f; break;
            case SAD:
                // Inner brows high and close, outer brows dropped: the classic
                // inverted-V sadness cue translated into the canine fur ridge.
                outerY = 298f; innerY = 216f; arch = 12f; strength = 1f; break;
            case CONCERNED:
                outerY = 284f; innerY = 226f; arch = 15f; strength = 1f; break;
            case ANNOYED:
                outerY = 236f; innerY = 298f; arch = 10f; strength = 1f; break;
            case ERROR:
                // A technical failure must not look like anger at the owner.
                outerY = 282f; innerY = 232f;
                arch = 14f; strength = 0.96f; break;
            default:
                break;
        }

        // Intensity selectively exaggerates the diagnostic brow silhouette;
        // pupil and idle motion are not amplified globally. Even a soft state
        // remains above the across-room readability floor.
        float browWeight = 0.64f + 0.36f * clamp01(affectIntensity);
        if (affect == Emotion.IDLE) browWeight = 1f;
        outerY = lerp(270f, outerY, browWeight);
        innerY = lerp(260f, innerY, browWeight);
        arch = lerp(15f, arch, browWeight);
        strength = lerp(0.58f, strength, browWeight);

        long activityElapsed = Math.max(0L, now - expressionStarted);
        long affectElapsed = Math.max(0L, now - affectStarted);
        float bounce = 0f;
        if (activity == Emotion.LISTENING) {
            float wakeLift = activityElapsed < 720L ?
                    7f * (1f - smoothStep(activityElapsed / 720f)) : 0f;
            outerY -= 5f + wakeLift;
            innerY -= 8f + wakeLift;
            arch += 5f;
            strength = Math.max(strength, 0.94f);
        } else if (affect == Emotion.IDLE && activity == Emotion.THINKING) {
            // Brows may act asymmetrically; the optical system may not.
            if (left) {
                outerY -= 18f; innerY -= 15f; arch += 12f;
            } else {
                outerY += 6f; innerY += 5f; arch = Math.max(8f, arch - 5f);
            }
            strength = Math.max(strength, 0.94f);
        } else if (affect == Emotion.IDLE && activity == Emotion.SPEAKING) {
            // Speech remains alive through pupil/lid punctuation. Moving the
            // entire eyebrow every syllable destroys the held emotional pose.
            strength = Math.max(strength, 0.72f);
        }
        if (affect == Emotion.LAUGHING && affectElapsed < 3200L) {
            bounce += 5f * laughingPulse(now);
        } else if (affect == Emotion.ERROR) {
            bounce += 3f * errorPulse(now);
        }
        outerY += bounce;
        innerY += bounce;

        int poseOffset = left ? 0 : 4;
        outerY = lerp(fromBrowPose[poseOffset], outerY, transition);
        innerY = lerp(fromBrowPose[poseOffset + 1], innerY, transition);
        arch = Math.max(4f, lerp(fromBrowPose[poseOffset + 2], arch, transition));
        strength = Math.max(0.55f,
                clamp01(lerp(fromBrowPose[poseOffset + 3], strength, transition)));
        synchronized (this) {
            displayedBrowPose[poseOffset] = outerY;
            displayedBrowPose[poseOffset + 1] = innerY;
            displayedBrowPose[poseOffset + 2] = arch;
            displayedBrowPose[poseOffset + 3] = strength;
        }

        Bitmap texture = left ? browLeftTexture : browRightTexture;
        if (texture == null) return;
        float outerX = x(left, 162f);
        float innerX = x(left, 418f);
        float width = 290f + 20f * strength;
        float halfHeight = 0.5f * (70f + 30f * strength);
        float centerX = 0.5f * (outerX + innerX);
        float geometryScale = width / Math.max(1f, Math.abs(innerX - outerX));
        for (int column = 0; column <= BROW_MESH_COLUMNS; column++) {
            float t = browMeshT[column];
            // Each atlas half already has the anatomically correct handedness:
            // left texture runs outer->inner, right texture inner->outer.
            float baseX = left ? lerp(outerX, innerX, t) :
                    lerp(innerX, outerX, t);
            float baseY = left ? lerp(outerY, innerY, t) :
                    lerp(innerY, outerY, t);
            baseX = centerX + (baseX - centerX) * geometryScale;
            float curveY = baseY - arch * 0.32f * browMeshArc[column];
            float taper = browMeshTaper[column];
            for (int row = 0; row <= BROW_MESH_ROWS; row++) {
                int vertex = (row * (BROW_MESH_COLUMNS + 1) + column) * 2;
                browMesh[vertex] = baseX;
                browMesh[vertex + 1] = curveY +
                        (row == 0 ? -halfHeight : halfHeight) * taper;
            }
        }
        // Mesh warping keeps the generated fur strands but makes the ridge
        // bend as anatomy rather than sliding a rigid eyebrow sticker. It also
        // guarantees mirror-equal handedness; the former right brow angle was
        // approximately 180 degrees and rendered its baked light upside-down.
        browTexturePaint.setAlpha((int)(218f + 37f * strength));
        canvas.drawBitmapMesh(texture, BROW_MESH_COLUMNS, BROW_MESH_ROWS,
                browMesh, 0, null, 0, browTexturePaint);
    }

    private void drawLids(Canvas canvas, boolean left, float closure,
                          Emotion activity, Emotion affect) {
        closure = clamp01(closure);
        if (closure < 0.006f || closedFur == null) return;

        float upperProgress = closure;
        float lowerProgress = closure;
        if (affect == Emotion.HAPPY || affect == Emotion.AFFECTIONATE) {
            // Smile with the lower lid. Closing both lids equally reads as
            // sleepy, not joyful, on this canine aperture.
            upperProgress = closure * 0.18f;
            lowerProgress = closure * 2.05f;
        } else if (affect == Emotion.LAUGHING) {
            // Laughter is a gesture, not a static smile: both lids squeeze,
            // with the lower lid still leading enough to retain the smile.
            upperProgress = closure * 0.86f;
            lowerProgress = closure * 1.58f;
        } else if (affect == Emotion.SLEEPY) {
            upperProgress = closure * 1.18f;
            lowerProgress = closure * 0.42f;
        } else if (affect == Emotion.SAD) {
            upperProgress = closure * 0.92f;
            lowerProgress = closure * 0.62f;
        } else if (affect == Emotion.CONCERNED) {
            upperProgress = closure * 0.72f;
            lowerProgress = closure * 0.72f;
        } else if (affect == Emotion.ANNOYED) {
            // Anger/annoyance reads from AU4 plus the lower-lid tightener, not
            // from a sleepy upper shutter.
            upperProgress = closure * 0.72f;
            lowerProgress = closure * 1.24f;
        } else if (affect == Emotion.ERROR) {
            upperProgress = closure * 0.82f;
            lowerProgress = closure * 0.78f;
        } else if (activity == Emotion.THINKING) {
            upperProgress = closure * 1.05f;
            lowerProgress = closure * 0.78f;
        }
        // Lids ride subtly over the globe during vertical gaze. Both sides use
        // the same displacement, preserving the binocular silhouette.
        if (gazeY > 0f) {
            upperProgress += gazeY * 0.10f;
            lowerProgress -= gazeY * 0.035f;
        } else {
            upperProgress += gazeY * 0.07f;
            lowerProgress -= gazeY * 0.045f;
        }
        float fullMix = smoothStep(clamp01((closure - 0.78f) / 0.22f));
        upperProgress = lerp(clamp01(upperProgress), closure, fullMix);
        lowerProgress = lerp(clamp01(lowerProgress), closure, fullMix);

        float upperSlope = targetUpperLidSlope(activity, affect, left);
        float lowerSlope = targetLowerLidSlope(activity, affect, left);
        buildAperture(left);
        // Emotion slope is a resting shape cue, not a diagonal full blink.
        upperSlope *= 1f - fullMix;
        lowerSlope *= 1f - fullMix;
        buildUpperCover(left, upperProgress, upperSlope);
        buildLowerCover(left, lowerProgress, lowerSlope);

        int eyeClip = canvas.save();
        canvas.clipPath(eyePath);

        int upperClip = canvas.save();
        canvas.clipPath(upperCoverPath);
        canvas.drawBitmap(closedFur, null, assetRect, bitmapPaint);
        canvas.restoreToCount(upperClip);

        int lowerClip = canvas.save();
        canvas.clipPath(lowerCoverPath);
        canvas.drawBitmap(closedFur, null, assetRect, bitmapPaint);
        canvas.restoreToCount(lowerClip);

        lidEdgePaint.setAlpha(180);
        canvas.drawPath(upperBoundaryPath, lidEdgePaint);
        canvas.drawPath(lowerBoundaryPath, lidEdgePaint);
        if (lowerProgress > 0.08f && closure < 0.96f) {
            canvas.drawPath(lowerBoundaryPath, wetEdgePaint);
        }
        canvas.restoreToCount(eyeClip);
    }

    private void updateBlink(long now, Emotion activity, Emotion affect) {
        if (blinkStarted >= 0) {
            long total = blinkCloseMs + blinkHoldMs + blinkOpenMs;
            if (now - blinkStarted >= total) {
                blinkStarted = -1;
                scheduleNextBlink(now, activity, affect);
            }
        }
        if (blinkStarted >= 0) return;

        if (queuedBlinkAt >= 0 && now >= queuedBlinkAt) {
            queuedBlinkAt = -1;
            startBlink(now, 0.94f, 118, 16, 244);
            return;
        }
        if (queuedSecondBlinkAt >= 0 && now >= queuedSecondBlinkAt) {
            queuedSecondBlinkAt = -1;
            startBlink(now, 0.84f, 108, 14, 226);
            return;
        }
        if (now >= nextBlink) startRandomBlink(now, activity, affect);
    }

    private void startRandomBlink(long now, Emotion activity, Emotion affect) {
        // A normal blink is usually a complete, fast binocular closure. Small
        // partial lid dips remain as secondary life, not as the dominant blink.
        boolean partial = random.nextFloat() < 0.28f;
        if (affect == Emotion.SLEEPY) {
            float depth = partial ? 0.58f + random.nextFloat() * 0.22f : 1f;
            startBlink(now, depth, 185 + random.nextInt(55),
                    240 + random.nextInt(320), 280 + random.nextInt(80));
        } else if (partial) {
            startBlink(now, 0.42f + random.nextFloat() * 0.22f,
                    86 + random.nextInt(25), 7 + random.nextInt(9),
                    174 + random.nextInt(58));
        } else {
            startBlink(now, 1f, 108 + random.nextInt(29),
                    12 + random.nextInt(15), 224 + random.nextInt(77));
        }
    }

    private void startBlink(long now, float depth, int closeMs, int holdMs, int openMs) {
        blinkStarted = now;
        blinkDepth = depth;
        blinkCloseMs = closeMs;
        blinkHoldMs = holdMs;
        blinkOpenMs = openMs;
    }

    private float blinkAmount(long now) {
        if (blinkStarted < 0) return 0f;
        long elapsed = now - blinkStarted;
        if (elapsed < blinkCloseMs) {
            // Endpoint-safe C2 easing avoids a velocity snap into the hold.
            float phase = clamp01(elapsed / (float) blinkCloseMs);
            return blinkDepth * smootherStep(phase);
        }
        elapsed -= blinkCloseMs;
        if (elapsed < blinkHoldMs) return blinkDepth;
        elapsed -= blinkHoldMs;
        if (elapsed < blinkOpenMs) {
            // Reopening remains slower through duration, while the curve also
            // leaves the hold and reaches rest with zero boundary velocity.
            float phase = clamp01(elapsed / (float) blinkOpenMs);
            return blinkDepth * (1f - smootherStep(phase));
        }
        return 0f;
    }

    private void scheduleNextBlink(long now, Emotion activity, Emotion affect) {
        int low;
        int spread;
        if (affect == Emotion.SLEEPY) {
            low = 2200; spread = 3200;
            nextBlink = now + low + random.nextInt(spread + 1);
            return;
        }
        switch (activity) {
            case LISTENING:
                low = 4200; spread = 5000; break;
            case THINKING:
                low = 3500; spread = 4300; break;
            case SPEAKING:
                low = 3600; spread = 4800; break;
            case IDLE:
            default:
                switch (affect) {
                    case HAPPY:
                    case AFFECTIONATE:
                    case LAUGHING:
                        low = 3200; spread = 4200; break;
                    case CURIOUS:
                    case SURPRISED:
                        low = 4200; spread = 4300; break;
                    case SAD:
                    case CONCERNED:
                        low = 4300; spread = 4800; break;
                    case ERROR:
                    case ANNOYED:
                        low = 6000; spread = 4200; break;
                    default:
                        low = 3800; spread = 6200; break;
                }
                break;
        }
        nextBlink = now + low + random.nextInt(spread + 1);
    }

    public synchronized void setTargetGaze(float x, float y) {
        if (!Float.isFinite(x) || !Float.isFinite(y) || Math.abs(x)>1f || Math.abs(y)>1f)
            throw new IllegalArgumentException("invalid gaze");
        gazeFromX = gazeX;
        gazeFromY = gazeY;
        gazeTargetX = x * .58f;
        gazeTargetY = y * .50f;
        gazeStarted = SystemClock.uptimeMillis();
        gazeDurationMs = 300;
        targetGazeUntil = gazeStarted + 1800L;
        postInvalidateOnAnimation();
    }

    private void updateGaze(long now, Emotion activity, Emotion affect) {
        long elapsed = now - gazeStarted;
        if (elapsed < gazeDurationMs) {
            // Quintic easing gives zero velocity and acceleration at both ends.
            // Combined with vsync scheduling this removes the visible two-step
            // arrival that the old 42-150 ms saccades produced.
            float amount = smootherStep(elapsed / (float) gazeDurationMs);
            gazeX = lerp(gazeFromX, gazeTargetX, amount);
            gazeY = lerp(gazeFromY, gazeTargetY, amount);
        } else {
            gazeX = gazeTargetX;
            gazeY = gazeTargetY;
        }

        if (now < targetGazeUntil) return;
        if (activity != gazeActivity || affect != gazeAffect) {
            // A new conversational/emotional state gets an immediate, smooth
            // attention decision.  Reset only the motif counters; the current
            // position is carried forward so no state change can pop the eyes.
            gazeActivity = activity;
            gazeAffect = affect;
            gazeDecisionSequence = 0L;
            idleReturnToContactNext = false;
            nextGaze = now;
        }
        if (now < nextGaze) return;
        gazeFromX = gazeX;
        gazeFromY = gazeY;
        gazeStarted = now;
        int hold;
        long affectElapsed = Math.max(0L, now - affectStarted);
        boolean canonicalHold = affect != Emotion.IDLE && affectElapsed < 1500L;
        if (activity == Emotion.IDLE && affect != Emotion.IDLE) {
            // Emotion owns gaze. The first 1.5 seconds are a clean canonical
            // hold, then only sparse, state-consistent accents are allowed.
            float jitterX = canonicalHold ? 0f : signedRandom(0.055f);
            float jitterY = canonicalHold ? 0f : signedRandom(0.040f);
            switch (affect) {
                case HAPPY:
                    gazeTargetX = jitterX; gazeTargetY = -0.10f + jitterY;
                    hold = canonicalHold ? 1700 : 1250 + random.nextInt(1250); break;
                case LAUGHING:
                    gazeTargetX = jitterX; gazeTargetY = -0.13f + jitterY;
                    hold = canonicalHold ? 1650 : 700 + random.nextInt(700); break;
                case CURIOUS:
                    gazeTargetX = jitterX;
                    gazeTargetY = -0.18f + jitterY;
                    hold = canonicalHold ? 1800 : 1500 + random.nextInt(1200); break;
                case SURPRISED:
                    gazeTargetX = jitterX * 0.25f; gazeTargetY = jitterY * 0.25f;
                    hold = canonicalHold ? 1900 : 1650 + random.nextInt(1200); break;
                case AFFECTIONATE:
                    gazeTargetX = jitterX * 0.45f; gazeTargetY = -0.08f + jitterY * 0.45f;
                    hold = canonicalHold ? 2000 : 1900 + random.nextInt(1500); break;
                case SLEEPY:
                    gazeTargetX = jitterX; gazeTargetY = 0.42f + jitterY;
                    hold = canonicalHold ? 1900 : 2400 + random.nextInt(2100); break;
                case SAD:
                    gazeTargetX = ((gazeDecisionSequence & 1L) == 0L ? 0.16f : -0.16f) +
                            jitterX;
                    gazeTargetY = 0.43f + jitterY;
                    hold = canonicalHold ? 2100 : 1800 + random.nextInt(1700); break;
                case CONCERNED:
                    gazeTargetX = canonicalHold ? 0f :
                            ((gazeDecisionSequence & 1L) == 0L ? 0.18f : -0.18f);
                    gazeTargetY = -0.04f + jitterY;
                    hold = canonicalHold ? 1850 : 1350 + random.nextInt(1050); break;
                case ANNOYED:
                    gazeTargetX = canonicalHold ? 0f :
                            ((gazeDecisionSequence & 1L) == 0L ? 0.22f : -0.22f);
                    gazeTargetY = 0.08f + jitterY * 0.35f;
                    hold = canonicalHold ? 2050 : 2100 + random.nextInt(1300); break;
                case ERROR:
                    gazeTargetX = 0f; gazeTargetY = 0f;
                    hold = canonicalHold ? 1850 : 2300 + random.nextInt(1700); break;
                default:
                    gazeTargetX = jitterX; gazeTargetY = jitterY;
                    hold = 1500 + random.nextInt(1300); break;
            }
        } else {
            switch (activity) {
                case LISTENING:
                    // Mutual regard owns four out of five fixations.  The fifth
                    // is a short thinking glance, followed by direct contact.
                    if (gazeDecisionSequence % 5L == 4L) {
                        gazeTargetX = ((gazeDecisionSequence / 5L) & 1L) == 0L
                                ? -0.30f : 0.30f;
                        gazeTargetY = -0.09f;
                        hold = 520 + random.nextInt(280);
                    } else {
                        gazeTargetX = signedRandom(0.075f);
                        gazeTargetY = -0.025f + signedRandom(0.045f);
                        hold = 900 + random.nextInt(650);
                    }
                    break;
                case THINKING:
                    // A deliberate three-point thought loop reads as searching
                    // memory instead of staring forever into one upper corner.
                    switch ((int)(gazeDecisionSequence % 3L)) {
                        case 0:
                            gazeTargetX = -0.44f; gazeTargetY = -0.30f;
                            hold = 720 + random.nextInt(430); break;
                        case 1:
                            gazeTargetX = 0.40f; gazeTargetY = -0.24f;
                            hold = 620 + random.nextInt(380); break;
                        default:
                            gazeTargetX = signedRandom(0.065f);
                            gazeTargetY = signedRandom(0.045f);
                            hold = 1050 + random.nextInt(550); break;
                    }
                    break;
                case SPEAKING:
                    // Mostly address the person; a brief phrase-boundary glance
                    // every fifth decision gives life without looking evasive.
                    if (gazeDecisionSequence % 5L != 4L) {
                        gazeTargetX = signedRandom(0.095f);
                        gazeTargetY = signedRandom(0.060f);
                        hold = 850 + random.nextInt(750);
                    } else {
                        gazeTargetX = ((gazeDecisionSequence / 5L) & 1L) == 0L
                                ? -0.32f : 0.32f;
                        gazeTargetY = -0.08f;
                        hold = 480 + random.nextInt(300);
                    }
                    break;
                case IDLE:
                default:
                    // FIXATE -> readable glance -> FIXATE.  A coprime stride
                    // visits all eight directional targets before repeating,
                    // while a soft near-centre return prevents robotic sweeps.
                    if (idleReturnToContactNext) {
                        gazeTargetX = signedRandom(0.060f);
                        gazeTargetY = signedRandom(0.040f);
                        hold = 2400 + random.nextInt(1400);
                        idleReturnToContactNext = false;
                    } else {
                        float[] target = AMBIENT_GAZE_TARGETS[ambientGazeIndex];
                        gazeTargetX = target[0] + signedRandom(0.025f);
                        gazeTargetY = target[1] + signedRandom(0.018f);
                        ambientGazeIndex = (ambientGazeIndex + 1) %
                                AMBIENT_GAZE_TARGETS.length;
                        hold = 650 + random.nextInt(400);
                        idleReturnToContactNext = true;
                    }
                    break;
            }
        }
        // With no animated head to share large gaze shifts, extreme eye-only
        // travel reads as divergence. Preserve a broad, readable range while
        // keeping the iris inside its canine aperture.
        gazeTargetX = clamp(gazeTargetX, -0.58f, 0.58f);
        gazeTargetY = clamp(gazeTargetY, -0.46f, 0.46f);
        float deltaX = gazeTargetX - gazeFromX;
        float deltaY = gazeTargetY - gazeFromY;
        float distance = (float)Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        // Continuous distance-to-duration mapping avoids a speed discontinuity
        // at an arbitrary threshold.  The 13-29 frame range is intentionally
        // slower than a biological saccade: this is a stylized display, where
        // the owner should see a fluid glance rather than a one-frame jump.
        float travel = smootherStep(clamp01(distance / 0.78f));
        gazeDurationMs = Math.round(220f + 260f * travel);
        // ``hold`` describes a readable fixation after the eyes arrive. The
        // old start-to-start deadline consumed up to 400 ms of that hold while
        // the gaze was still travelling and could look like continuous drift.
        nextGaze = now + gazeDurationMs + hold;
        gazeDecisionSequence++;
    }

    private void updateAmbientVisual(long now, Emotion activity, Emotion affect) {
        boolean eligible = activity == Emotion.IDLE && affect == Emotion.IDLE;
        if (!eligible) {
            if (ambientIdleEligible) {
                // A spoken command must take ownership immediately, but the
                // last few percent of the idle mouth/brow pose fade away over
                // several display frames instead of snapping on frame one.
                if (ambientVisualAffect != Emotion.IDLE &&
                        ambientVisualBlend > 0.002f) {
                    ambientVisualExitStarted = now;
                    ambientVisualExitFromBlend = ambientVisualBlend;
                } else {
                    clearAmbientVisual();
                }
                ambientVisualStarted = -1L;
            }
            ambientIdleEligible = false;
            if (ambientVisualExitStarted >= 0L) {
                float progress = (now - ambientVisualExitStarted) /
                        (float)AMBIENT_VISUAL_EXIT_MS;
                if (progress >= 1f) {
                    clearAmbientVisual();
                } else {
                    ambientVisualBlend = ambientVisualExitFromBlend *
                            (1f - smootherStep(progress));
                }
            }
            return;
        }

        if (!ambientIdleEligible) {
            ambientIdleEligible = true;
            clearAmbientVisual();
            // The face first establishes a calm baseline, then begins its
            // private idle thought rhythm at an irregular moment.
            nextAmbientVisual = now + 20_000L + random.nextInt(40_001);
            return;
        }

        if (ambientVisualStarted < 0L) {
            // Owner 08.10: in idle one calm waiting face; feelings only for real
            // events (the mind sends them). The random micro-moods every 20-60 s,
            // «annoyed» among them, made the real ones unreadable (audit 09.10).
            // Blinks and gaze stay alive; the roulette below is switched off.
            if (!IDLE_MICRO_MOODS || now < nextAmbientVisual) return;
            int index = random.nextInt(IDLE_MICRO_AFFECTS.length);
            if (index == lastAmbientVisualIndex) {
                index = (index + 1 + random.nextInt(
                        IDLE_MICRO_AFFECTS.length - 1)) %
                        IDLE_MICRO_AFFECTS.length;
            }
            lastAmbientVisualIndex = index;
            ambientVisualAffect = IDLE_MICRO_AFFECTS[index];
            ambientVisualDurationMs = 4_000 + random.nextInt(2_401);
            ambientVisualStarted = now;
            ambientVisualExitStarted = -1L;
            // Start-to-start cadence, not gap-after-finish cadence: a complete
            // idle expression happens only once every 20-60 seconds.
            nextAmbientVisual = now + 20_000L + random.nextInt(40_001);
        }

        float phase = (now - ambientVisualStarted) /
                (float)Math.max(1, ambientVisualDurationMs);
        if (phase >= 1f) {
            ambientVisualAffect = Emotion.IDLE;
            ambientVisualBlend = 0f;
            ambientVisualStarted = -1L;
            // A long screen/app suspension may skip past both the expression
            // and its next start deadline.  Do not replay missed idle moods in
            // a burst after resume; establish a fresh quiet 20-60 s interval.
            if (nextAmbientVisual <= now) {
                nextAmbientVisual = now + 20_000L + random.nextInt(40_001);
            }
            return;
        }

        // A long C2 fade-in and fade-out makes the face seem to drift into a
        // thought or mood. The low strength deliberately avoids a full emoji.
        float envelope;
        if (phase < 0.32f) {
            envelope = smootherStep(phase / 0.32f);
        } else if (phase > 0.68f) {
            envelope = smootherStep((1f - phase) / 0.32f);
        } else {
            envelope = 1f;
        }
        ambientVisualBlend = envelope * ambientVisualStrength(ambientVisualAffect);
    }

    private static float ambientVisualStrength(Emotion affect) {
        switch (affect) {
            case HAPPY: return 0.27f;
            case CURIOUS: return 0.25f;
            case SURPRISED: return 0.15f;
            case ANNOYED: return 0.12f;
            default: return 0f;
        }
    }

    private void clearAmbientVisual() {
        ambientVisualAffect = Emotion.IDLE;
        ambientVisualBlend = 0f;
        ambientVisualStarted = -1L;
        ambientVisualExitStarted = -1L;
        ambientVisualExitFromBlend = 0f;
    }

    private boolean isAmbientVisualMoving(long now) {
        if (ambientVisualExitStarted >= 0L) {
            return now - ambientVisualExitStarted < AMBIENT_VISUAL_EXIT_MS;
        }
        if (ambientVisualStarted < 0L) return false;
        float phase = (now - ambientVisualStarted) /
                (float)Math.max(1, ambientVisualDurationMs);
        // The middle hold is a genuinely static pose. Sleeping until the
        // fade-out boundary avoids redrawing two full face bitmaps at 60 fps.
        return phase < 0.32f || phase > 0.68f;
    }

    private void cancelStaticWake() {
        if (!staticWakeScheduled) return;
        removeCallbacks(staticWakeRunnable);
        staticWakeScheduled = false;
        staticWakeDeadline = 0L;
    }

    private void scheduleStaticWake(long now) {
        long delay = nextStaticWakeDelay(now);
        long deadline = now + delay;
        if (staticWakeScheduled && staticWakeDeadline == deadline) return;
        cancelStaticWake();
        staticWakeScheduled = true;
        staticWakeDeadline = deadline;
        postDelayed(staticWakeRunnable, delay);
    }

    @Override protected void onDetachedFromWindow() {
        cancelStaticWake();
        super.onDetachedFromWindow();
    }

    /**
     * When the pose is static, wake exactly for the next scheduled event rather
     * than repainting the entire textured face at 15 fps. Any setter still
     * invalidates immediately.
     */
    private long nextStaticWakeDelay(long now) {
        long deadline = Long.MAX_VALUE;
        if (nextBlink > now) deadline = Math.min(deadline, nextBlink);
        if (nextGaze > now) deadline = Math.min(deadline, nextGaze);
        if (queuedBlinkAt > now) deadline = Math.min(deadline, queuedBlinkAt);
        if (queuedSecondBlinkAt > now) deadline = Math.min(deadline, queuedSecondBlinkAt);
        if (ambientIdleEligible && ambientVisualStarted < 0L &&
                nextAmbientVisual > now) {
            deadline = Math.min(deadline, nextAmbientVisual);
        }
        if (ambientIdleEligible && ambientVisualStarted >= 0L) {
            long fadeOutStarts = ambientVisualStarted +
                    Math.round(ambientVisualDurationMs * 0.68f);
            if (fadeOutStarts > now) deadline = Math.min(deadline, fadeOutStarts);
        }
        synchronized (this) {
            if (interactionExpiresAt > now) deadline = Math.min(deadline, interactionExpiresAt);
            if (eventExpiresAt > now) deadline = Math.min(deadline, eventExpiresAt);
            if (reactionExpiresAt > now) deadline = Math.min(deadline, reactionExpiresAt);
            if (directExpiresAt > now) deadline = Math.min(deadline, directExpiresAt);
        }
        if (deadline == Long.MAX_VALUE) return 1_000L;
        return Math.max(1L, deadline - now);
    }

    private float signedRandom(float extent) {
        return (random.nextFloat() * 2f - 1f) * extent;
    }

    private float targetClosure(Emotion activity, Emotion affect, long now) {
        float base;
        switch (affect) {
            case HAPPY:
                // AU6-like lower-lid/cheek compression is the only strong
                // eye-only happiness cue, so it is intentionally clear.
                base = 0.52f; break;
            case LAUGHING:
                long laughElapsed = Math.max(0L, now - affectStarted);
                if (laughElapsed < 3100L) {
                    float pulse = 0.5f + 0.5f * laughingPulse(now);
                    float damping = 1f - 0.18f * clamp01(laughElapsed / 3100f);
                    base = 0.55f + 0.44f * pulse * damping;
                } else {
                    base = 0.72f;
                }
                break;
            case CURIOUS:
                base = 0.015f; break;
            case SURPRISED:
                base = 0f; break;
            case AFFECTIONATE:
                base = 0.23f; break;
            case SLEEPY:
                base = 0.86f; break;
            case SAD:
                base = 0.31f; break;
            case CONCERNED:
                base = 0.12f; break;
            case ANNOYED:
                base = 0.62f; break;
            case ERROR:
                float settle = smoothStep(clamp01((now - transitionStarted - 90f) / 180f));
                base = (0.15f + 0.05f * (0.5f + 0.5f * errorPulse(now))) * settle;
                break;
            default:
                base = 0.035f; break;
        }

        if (affect != Emotion.IDLE) {
            float closureWeight = 0.66f + 0.34f * clamp01(affectIntensity);
            base = lerp(0.035f, base, closureWeight);
        }

        if (activity == Emotion.LISTENING) {
            // Open-eyed mutual attention overlays every affect. The prior
            // synthetic 88% wake blink hid the acknowledgement itself.
            base = Math.min(0.11f, base * 0.42f);
        } else if (activity == Emotion.THINKING) {
            base = Math.min(0.92f, base + 0.12f);
        } else if (activity == Emotion.SPEAKING) {
            float pulse = 0.5f + 0.5f * speakingPulse(now);
            base = Math.min(0.92f, base * 0.90f + 0.018f + 0.032f * pulse);
        }
        return clamp01(base);
    }

    private float targetPupil(Emotion activity, Emotion affect, long now) {
        float pupil;
        switch (affect) {
            case HAPPY: pupil = 0.58f; break;
            case LAUGHING: pupil = 0.55f + 0.04f * laughingPulse(now); break;
            case CURIOUS: pupil = 0.62f; break;
            case SURPRISED: pupil = 0.36f; break;
            case AFFECTIONATE: pupil = 0.64f; break;
            case SLEEPY: pupil = 0.38f; break;
            case SAD: pupil = 0.61f; break;
            case CONCERNED: pupil = 0.59f; break;
            case ANNOYED: pupil = 0.29f; break;
            case ERROR: pupil = 0.48f; break;
            default: pupil = 0.48f; break;
        }
        if (activity == Emotion.LISTENING) {
            long elapsed = Math.max(0L, now - expressionStarted);
            float attention = smoothStep(clamp01(elapsed / 170f));
            float settle = smoothStep(clamp01((elapsed - 260f) / 460f));
            pupil = lerp(pupil, Math.max(pupil, 0.64f), attention * 0.72f);
            pupil = lerp(pupil, Math.min(0.62f, pupil), settle * 0.20f);
        } else if (activity == Emotion.THINKING) {
            pupil -= 0.08f;
        } else if (activity == Emotion.SPEAKING) {
            pupil += 0.025f * speakingPulse(now);
        }
        return Math.max(0.22f, Math.min(0.72f, pupil));
    }

    private float targetIrisScale(Emotion activity, Emotion affect, long now) {
        float scale;
        switch (affect) {
            case HAPPY: scale = 1.025f; break;
            case LAUGHING: scale = 1.03f; break;
            case CURIOUS: scale = 1.045f; break;
            // A smaller iris exposes more of the aperture and approximates
            // upper-lid raise/AU5 on this fixed dog-eye shell.
            case SURPRISED: scale = 0.94f; break;
            case AFFECTIONATE: scale = 1.04f; break;
            case SLEEPY: scale = 0.97f; break;
            case SAD: scale = 1.035f; break;
            case CONCERNED: scale = 1.035f; break;
            case ERROR: scale = 0.99f; break;
            case ANNOYED: scale = 0.96f; break;
            default: scale = 1f; break;
        }
        if (activity == Emotion.LISTENING) {
            long elapsed = Math.max(0L, now - expressionStarted);
            float attention = smoothStep(clamp01(elapsed / 260f));
            float settle = smoothStep(clamp01((elapsed - 260f) / 460f));
            scale += lerp(0f, 0.018f, attention) - 0.010f * settle;
        } else if (activity == Emotion.THINKING) {
            scale -= 0.03f;
        }
        return Math.max(0.94f, Math.min(1.08f, scale));
    }

    private float targetUpperLidSlope(Emotion activity, Emotion affect, boolean left) {
        float canonical;
        switch (affect) {
            case SAD: canonical = -0.34f; break;
            case CONCERNED: canonical = -0.25f; break;
            case ANNOYED: canonical = 0.32f; break;
            case ERROR: canonical = 0.16f; break;
            case CURIOUS: canonical = -0.12f; break;
            default: canonical = 0f; break;
        }
        if (affect == Emotion.IDLE && activity == Emotion.THINKING) {
            canonical -= 0.10f;
        }
        return left ? canonical : -canonical;
    }

    private float targetLowerLidSlope(Emotion activity, Emotion affect, boolean left) {
        float canonical = 0f;
        if (affect == Emotion.HAPPY || affect == Emotion.AFFECTIONATE ||
                affect == Emotion.LAUGHING) canonical = 0.22f;
        else if (affect == Emotion.ANNOYED) canonical = 0.20f;
        return left ? canonical : -canonical;
    }

    private void buildGlobeClip(boolean left) {
        eyePath.rewind();
        float outer = x(left, 150f);
        float inner = x(left, 434f);
        eyePath.moveTo(outer, 390f);
        eyePath.cubicTo(x(left, 181f), 298f, x(left, 300f), 270f, inner, 386f);
        eyePath.cubicTo(x(left, 409f), 462f, x(left, 221f), 489f, outer, 390f);
        eyePath.close();
    }

    private void buildAperture(boolean left) {
        eyePath.rewind();
        float outer = x(left, 154f);
        float inner = x(left, 430f);
        eyePath.moveTo(outer, 390f);
        eyePath.cubicTo(x(left, 185f), 303f, x(left, 300f), 275f, inner, 386f);
        eyePath.cubicTo(x(left, 405f), 457f, x(left, 225f), 484f, outer, 390f);
        eyePath.close();
    }

    private void buildUpperCover(boolean left, float progress, float slope) {
        float outerX = x(left, 154f);
        float innerX = x(left, 430f);
        float c1X = x(left, lerp(185f, 225f, progress));
        float c2X = x(left, lerp(300f, 370f, progress));
        float finalOuterY = tiltedY(left, outerX, 399f, slope);
        float finalC1Y = tiltedY(left, c1X, 449f, slope);
        float finalC2Y = tiltedY(left, c2X, 454f, slope);
        float finalInnerY = tiltedY(left, innerX, 410f, slope);
        float outerY = lerp(390f, finalOuterY, progress);
        float c1Y = lerp(303f, finalC1Y, progress);
        float c2Y = lerp(275f, finalC2Y, progress);
        float innerY = lerp(386f, finalInnerY, progress);

        upperBoundaryPath.rewind();
        upperBoundaryPath.moveTo(outerX, outerY);
        upperBoundaryPath.cubicTo(c1X, c1Y, c2X, c2Y, innerX, innerY);

        upperCoverPath.rewind();
        upperCoverPath.moveTo(outerX, 390f);
        upperCoverPath.cubicTo(x(left, 185f), 303f, x(left, 300f), 275f, innerX, 386f);
        upperCoverPath.lineTo(innerX, innerY);
        upperCoverPath.cubicTo(c2X, c2Y, c1X, c1Y, outerX, outerY);
        upperCoverPath.close();
    }

    private void buildLowerCover(boolean left, float progress, float slope) {
        float outerX = x(left, 154f);
        float innerX = x(left, 430f);
        float c1X = x(left, lerp(225f, 225f, progress));
        float c2X = x(left, lerp(405f, 370f, progress));
        float finalOuterY = tiltedY(left, outerX, 399f, slope);
        float finalC1Y = tiltedY(left, c1X, 449f, slope);
        float finalC2Y = tiltedY(left, c2X, 454f, slope);
        float finalInnerY = tiltedY(left, innerX, 410f, slope);
        float outerY = lerp(390f, finalOuterY, progress);
        float c1Y = lerp(484f, finalC1Y, progress);
        float c2Y = lerp(457f, finalC2Y, progress);
        float innerY = lerp(386f, finalInnerY, progress);

        lowerBoundaryPath.rewind();
        lowerBoundaryPath.moveTo(outerX, outerY);
        lowerBoundaryPath.cubicTo(c1X, c1Y, c2X, c2Y, innerX, innerY);

        lowerCoverPath.rewind();
        lowerCoverPath.moveTo(innerX, 386f);
        lowerCoverPath.cubicTo(x(left, 405f), 457f, x(left, 225f), 484f, outerX, 390f);
        lowerCoverPath.lineTo(outerX, outerY);
        lowerCoverPath.cubicTo(c1X, c1Y, c2X, c2Y, innerX, innerY);
        lowerCoverPath.close();
    }

    private float tiltedY(boolean left, float screenX, float baseY, float slope) {
        float center = left ? LEFT_CENTER_X : RIGHT_CENTER_X;
        return baseY + (screenX - center) * slope;
    }

    private float x(boolean left, float leftEyeX) {
        return left ? leftEyeX : BASE_W - leftEyeX;
    }

    private static float smoothStep(float value) {
        value = clamp01(value);
        return value * value * (3f - 2f * value);
    }

    private static float smootherStep(float value) {
        value = clamp01(value);
        return value * value * value *
                (value * (value * 6f - 15f) + 10f);
    }

    /** Decisive readable accent followed by a settle, never a disappearing counter-pose. */
    private static float cartoonPoseProgress(float value) {
        value = clamp01(value);
        if (value < 0.68f) {
            return lerp(0f, 1.075f, smoothStep(value / 0.68f));
        }
        return lerp(1.075f, 1f, smoothStep((value - 0.68f) / 0.32f));
    }

    private float speakingPulse(long now) {
        float elapsed = Math.max(0L, now - expressionStarted);
        return (float)Math.sin(elapsed / 175.0);
    }

    private float laughingPulse(long now) {
        float elapsed = Math.max(0L, now - affectStarted);
        // Three strong binocular accents, then a stable happy hold.
        float envelope = 1f - smoothStep(clamp01((elapsed - 1450f) / 950f));
        return (float)Math.sin(elapsed / 92.0) * envelope;
    }

    private float errorPulse(long now) {
        float elapsed = Math.max(0L, now - affectStarted);
        // A short shared flinch; never independent per-eye glitches.
        float envelope = 1f - smoothStep(clamp01((elapsed - 360f) / 420f));
        return (float)Math.sin(elapsed / 68.0) * envelope;
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }

    private static float lerp(float from, float to, float amount) {
        return from + (to - from) * amount;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            downX = event.getX();
            downY = event.getY();
            return true;
        }
        if (event.getAction() == MotionEvent.ACTION_UP) {
            float dx = event.getX() - downX;
            float dy = event.getY() - downY;
            if (dx * dx + dy * dy < 2500f && event.getY() < getHeight() / 3f) {
                performClick();
            }
            return true;
        }
        return true;
    }

    @Override
    public boolean performClick() {
        super.performClick();
        Emotion next;
        synchronized (this) {
            next = Emotion.values()[(emotion.ordinal() + 1) % Emotion.values().length];
        }
        setEmotion(next.name());
        return true;
    }
}
