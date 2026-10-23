package ai.robotbrain.face;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.LinearGradient;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RadialGradient;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Shader;

/**
 * Production renderer built directly from the owner-approved corgi model sheet.
 *
 * <p>The eight key poses are exact normalized crops of that sheet.  Expression
 * changes use an animation-style blink cut: both eyes close, the exact target
 * pose is exchanged at peak closure, then both eyes reopen.  This preserves the
 * approved drawing at every held pose and avoids optical-flow double features.
 * Each source socket is cleaned once when the rig is created.  At runtime an
 * independent optical globe (sclera, limbus, iris, pupil and catchlights)
 * rotates behind the original pose-specific socket and lids.  One conjugate
 * gaze vector rotates both globes together.  This deliberately does not paste
 * a generic white eye over the artwork: the wet line, surrounding fur, brows
 * and expression remain the exact approved drawing.</p>
 */
final class ApprovedCorgiSpriteRig {
    private static final float BASE_W = 1080f;
    private static final float BASE_H = 720f;

    // Authoritative centres from MEIZU16TH_EYE_CUTOUTS.md. A 130 px radius
    // fits the complete 260 px-high legacy animated envelope while leaving a
    // clean OLED-black gap between the two circular windows.
    private static final float LEFT_PORTHOLE_X = 295f;
    private static final float RIGHT_PORTHOLE_X = 785f;
    private static final float PORTHOLE_Y = 380f;
    private static final float PORTHOLE_RADIUS = 130f;
    // Horizontal crop anchors sit slightly outside each painted eye. This
    // keeps the broad approved brow and enough orange/cream fur to read as a
    // face fragment while excluding the central nose and mouth.
    private static final float LEFT_SOURCE_X = 415f;
    private static final float RIGHT_SOURCE_X = 665f;

    private static final int NEUTRAL = 0;
    private static final int HAPPY = 1;
    private static final int LAUGHING = 2;
    private static final int SURPRISED = 3;
    private static final int SAD = 4;
    private static final int CONCERNED = 5;
    private static final int ANNOYED = 6;
    private static final int CURIOUS = 7;
    // Eyebrows occupy materially different vertical bands in the approved
    // drawings. A single zoom either cut those brows or admitted the nose at
    // the lower inner edge. Each exact pose therefore has its own fitted crop:
    // the complete brow, eyelids and optical opening remain inside the same
    // physical 260 px circle in idle and every emotion. These are presentation
    // transforms only; the source anatomy and shared gaze/blink stay intact.
    private static final float[] PORTHOLE_SOURCE_Y = {
            335f, 320f, 295f, 325f, 350f, 345f, 335f, 345f,
    };
    private static final float[] PORTHOLE_SCALE = {
            1.00f, 1.00f, 1.00f, 1.00f, 0.96f, 0.98f, 1.00f, 0.94f,
    };
    // A closing eyelid has to cover the stationary wet rim as well as the
    // optical opening. The moving open eye itself remains behind that original
    // rim, so the accepted drawing is preserved whenever the eye is open.
    private static final float BLINK_COVER_MARGIN = 5f;
    // Thickness of the painted wet rim around each aperture: the lid must cover it.
    private static final float LID_RIM = 9f;
    private static final float LID_TOP_RIM = 18f;
    private static final float LID_FEATHER = 8f;
    private static final float AFFECTIONATE_LIDS = 0.38f;
    private static final int LID_SAMPLES = 31;

    private static final int[] KEYFRAME_RESOURCES = {
            R.drawable.corgi_approved_neutral_v3,
            R.drawable.corgi_approved_happy_v3,
            R.drawable.corgi_approved_laughing_v3,
            R.drawable.corgi_approved_surprised_v3,
            R.drawable.corgi_approved_sad_v3,
            R.drawable.corgi_approved_concerned_v3,
            R.drawable.corgi_approved_annoyed_v3,
            R.drawable.corgi_approved_curious_v3,
    };

    // Pose-specific visible apertures, measured just inside each original dark
    // wet rim.  They are intentionally independent: annoyed and curious are
    // strongly asymmetric and cannot share one generic oval without looking
    // like pasted-on glasses.
    private static final float[] LEFT_SOCKET_X =
            {439.5f, 440.5f, 0f, 451f, 430.5f, 439f, 433f, 447f};
    private static final float[] RIGHT_SOCKET_X =
            {640.5f, 636.5f, 0f, 635.5f, 642f, 640f, 628.5f, 644.5f};
    private static final float[] LEFT_SOCKET_Y =
            {379.5f, 367.5f, 0f, 376.5f, 406f, 402f, 399f, 420f};
    private static final float[] RIGHT_SOCKET_Y =
            {380f, 369f, 0f, 380f, 406.5f, 401.5f, 403f, 389f};
    private static final float[] LEFT_SOCKET_RX =
            {48.5f, 47.5f, 0f, 45f, 53.5f, 49f, 65f, 52f};
    private static final float[] RIGHT_SOCKET_RX =
            {46.5f, 45.5f, 0f, 43.5f, 52f, 46f, 65.5f, 46.5f};
    private static final float[] LEFT_SOCKET_RY =
            {46.5f, 44.5f, 0f, 46.5f, 48f, 48f, 39f, 50f};
    private static final float[] RIGHT_SOCKET_RY =
            {47f, 45f, 0f, 47f, 50.5f, 48.5f, 36f, 50f};
    private final Bitmap[] keyframes = new Bitmap[KEYFRAME_RESOURCES.length];
    private final Bitmap[][] eyeTextures =
            new Bitmap[KEYFRAME_RESOURCES.length][2];
    private final RectF designRect = new RectF(0f, 0f, BASE_W, BASE_H);
    // Tight crops stop above the laughing pose's pale muzzle and stay outside
    // the white central blaze. Wider crops leaked those light pixels into the
    // lower arc of a fully closed sleepy eye.
    private final Rect leftClosedSource = new Rect(365, 325, 490, 400);
    private final Rect rightClosedSource = new Rect(580, 325, 705, 400);
    private final RectF featureDestination = new RectF();
    private final RectF apertureDestination = new RectF();
    private final Paint layerPaint = new Paint(
            Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint scleraPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint portholeEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RadialGradient scleraGradient;
    private final Matrix opticalGradientMatrix = new Matrix();
    private final Path eyeClip = new Path();
    private final Path upperLidClip = new Path();
    private final Path lowerLidClip = new Path();
    private final Path lidUnion = new Path();
    private final Path portholeClip = new Path();
    private final Rect templeSource = new Rect();
    private final Paint lidShadePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint lashPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint lidPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint lidMaskPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final ColorMatrixColorFilter[][] lidFilters = new ColorMatrixColorFilter[KEYFRAME_RESOURCES.length][2];
    private final Bitmap[][] lidMasks = new Bitmap[KEYFRAME_RESOURCES.length][2];

    private final boolean circularEyeOnly;

    private int fromPose = NEUTRAL;
    private int targetPose = NEUTRAL;
    private boolean initialized;
    private long transitionStarted;
    private long lastSequence = Long.MIN_VALUE;
    private float transitionDurationMs = 280f;
    private float transitionInitialClosure;
    private boolean transitionHoldClosed;
    private float presentationClosureTarget;
    private float lastPresentedClosure;
    private float speechAmount = -1f;
    private static final int JAW_COLUMNS = 72;
    private static final int JAW_ROWS = 144;
    private final float[] jawMesh = new float[(JAW_COLUMNS + 1) * (JAW_ROWS + 1) * 2];

    ApprovedCorgiSpriteRig(Context context, boolean circularEyeOnly) {
        this.circularEyeOnly = circularEyeOnly;
        Context appContext = context.getApplicationContext();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        for (int index = 0; index < keyframes.length; index++) {
            Bitmap decoded = decodeScaled(appContext, KEYFRAME_RESOURCES[index],
                    (int)BASE_W, (int)BASE_H, options);
            if (decoded != null && index != LAUGHING &&
                    LEFT_SOCKET_RX[index] > 0f) {
                eyeTextures[index][0] = buildEyeTexture(decoded, index, true);
                eyeTextures[index][1] = buildEyeTexture(decoded, index, false);
                prepareLid(decoded, index, true);
                prepareLid(decoded, index, false);
            }
            keyframes[index] = prepareFacePlate(decoded, index);
        }
        scleraGradient = new RadialGradient(0f, 0f, 1f,
                new int[]{
                        Color.rgb(255, 252, 235),
                        Color.rgb(250, 242, 214),
                        Color.rgb(218, 195, 157),
                },
                new float[]{0f, 0.72f, 1f}, Shader.TileMode.CLAMP);
        scleraPaint.setStyle(Paint.Style.FILL);
        scleraPaint.setShader(scleraGradient);
        // The black antialiased edge covers hardware clip quantisation without
        // adding a visible glasses-like frame around the fur islands.
        portholeEdgePaint.setStyle(Paint.Style.STROKE);
        portholeEdgePaint.setStrokeWidth(8f);
        portholeEdgePaint.setColor(Color.BLACK);
        lashPaint.setStyle(Paint.Style.STROKE);
        lashPaint.setStrokeCap(Paint.Cap.ROUND);
        lashPaint.setStrokeJoin(Paint.Join.ROUND);
        lashPaint.setColor(Color.rgb(38, 22, 14));
        lidShadePaint.setStyle(Paint.Style.FILL);
        lidMaskPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
    }

    private Bitmap decodeScaled(Context context, int resource, int width, int height,
                                BitmapFactory.Options options) {
        Bitmap decoded = BitmapFactory.decodeResource(
                context.getResources(), resource, options);
        if (decoded == null) return null;
        if (decoded.getWidth() == width && decoded.getHeight() == height) {
            return decoded;
        }
        Bitmap scaled = Bitmap.createScaledBitmap(decoded, width, height, true);
        if (scaled != decoded) decoded.recycle();
        return scaled;
    }

    /**
     * Preserve the approved optical artwork as one movable globe texture.
     * Only pixels inside the measured wet rim are retained; the original rim
     * and lids remain on the foreground face plate.  A soft alpha edge lets a
     * warm backing sclera fill newly exposed space during rotation without a
     * rectangular crop or a second painted eye appearing on top.
     */
    private Bitmap buildEyeTexture(Bitmap source, int pose, boolean left) {
        float centerX = left ? LEFT_SOCKET_X[pose] : RIGHT_SOCKET_X[pose];
        float centerY = left ? LEFT_SOCKET_Y[pose] : RIGHT_SOCKET_Y[pose];
        float radiusX = left ? LEFT_SOCKET_RX[pose] : RIGHT_SOCKET_RX[pose];
        float radiusY = left ? LEFT_SOCKET_RY[pose] : RIGHT_SOCKET_RY[pose];
        int cropLeft = Math.max(0, (int)Math.floor(centerX - radiusX));
        int cropTop = Math.max(0, (int)Math.floor(centerY - radiusY));
        int cropRight = Math.min(source.getWidth(),
                (int)Math.ceil(centerX + radiusX));
        int cropBottom = Math.min(source.getHeight(),
                (int)Math.ceil(centerY + radiusY));
        int width = Math.max(1, cropRight - cropLeft);
        int height = Math.max(1, cropBottom - cropTop);
        Bitmap texture = Bitmap.createBitmap(width, height,
                Bitmap.Config.ARGB_8888);
        texture.setHasAlpha(true);
        int[] pixels = new int[width * height];
        source.getPixels(pixels, 0, width, cropLeft, cropTop, width, height);
        float localCenterX = centerX - cropLeft;
        float localCenterY = centerY - cropTop;
        float feather = 1.7f / Math.max(1f, Math.min(radiusX, radiusY));
        for (int y = 0; y < height; y++) {
            float dy = (y + 0.5f - localCenterY) / radiusY;
            for (int x = 0; x < width; x++) {
                float dx = (x + 0.5f - localCenterX) / radiusX;
                float distance = (float)Math.sqrt(dx * dx + dy * dy);
                float keep = clamp01((1f - distance) / feather);
                int offset = y * width + x;
                int color = pixels[offset];
                pixels[offset] = Color.argb(
                        Math.round(Color.alpha(color) * keep),
                        Color.red(color), Color.green(color), Color.blue(color));
            }
        }
        texture.setPixels(pixels, 0, width, 0, 0, width, height);
        return texture;
    }

    /**
     * Convert an approved pose into a foreground face plate.  Only the exact
     * optical apertures become transparent; the original wet rim, eyelids,
     * brows, fur and muzzle stay untouched.  Complete moving eye globes are
     * rendered beneath this plate, so their edges can never sit on top of fur.
     */
    private Bitmap prepareFacePlate(Bitmap decoded, int pose) {
        if (decoded == null || pose == LAUGHING || LEFT_SOCKET_RX[pose] <= 0f) {
            return decoded;
        }
        Bitmap mutable = decoded.copy(Bitmap.Config.ARGB_8888, true);
        if (mutable == null) return decoded;
        if (mutable != decoded) decoded.recycle();
        // Resource PNGs without source transparency are decoded as opaque.
        // setPixels() cannot create a usable aperture until the Bitmap is
        // explicitly marked alpha-capable; without this flag Android silently
        // displayed the original stationary painted eyes over the whole rig.
        mutable.setHasAlpha(true);

        int width = mutable.getWidth();
        int height = mutable.getHeight();
        int[] pixels = new int[width * height];
        mutable.getPixels(pixels, 0, width, 0, 0, width, height);
        clearAperture(pixels, width, height,
                LEFT_SOCKET_X[pose], LEFT_SOCKET_Y[pose],
                LEFT_SOCKET_RX[pose], LEFT_SOCKET_RY[pose]);
        clearAperture(pixels, width, height,
                RIGHT_SOCKET_X[pose], RIGHT_SOCKET_Y[pose],
                RIGHT_SOCKET_RX[pose], RIGHT_SOCKET_RY[pose]);
        mutable.setPixels(pixels, 0, width, 0, 0, width, height);
        return mutable;
    }

    private static void clearAperture(int[] pixels, int width, int height,
                                      float centerX, float centerY,
                                      float radiusX, float radiusY) {
        int left = Math.max(0, (int)Math.floor(centerX - radiusX - 2f));
        int right = Math.min(width - 1, (int)Math.ceil(centerX + radiusX + 2f));
        int top = Math.max(0, (int)Math.floor(centerY - radiusY - 2f));
        int bottom = Math.min(height - 1, (int)Math.ceil(centerY + radiusY + 2f));
        for (int y = top; y <= bottom; y++) {
            float dy = (y + 0.5f - centerY) / radiusY;
            for (int x = left; x <= right; x++) {
                float dx = (x + 0.5f - centerX) / radiusX;
                float distance = (float)Math.sqrt(dx * dx + dy * dy);
                // Two-pixel supersampling-style feather keeps the original wet
                // rim antialiasing while making the interior fully transparent.
                float featherWidth = 2f / Math.max(1f, Math.min(radiusX, radiusY));
                float keep = clamp01((distance - (1f - featherWidth)) /
                        featherWidth);
                int offset = y * width + x;
                int color = pixels[offset];
                pixels[offset] = Color.argb(
                        Math.round(Color.alpha(color) * keep),
                        Color.red(color), Color.green(color), Color.blue(color));
            }
        }
    }

    void draw(Canvas canvas, String activity, String affect, float intensity,
              float blink, float gazeX, float gazeY, float ignoredIrisScale,
              float ignoredPupilScale, float emotionalClosure,
              String ambientAffect,
              float ambientBlend, long now, long activityStarted,
              long affectStarted, long sequence, float speechAmount) {
        this.speechAmount = speechAmount;
        String safeActivity = activity == null ? "idle" : activity;
        String safeAffect = affect == null ? "idle" : affect;
        String safeAmbientAffect = ambientAffect == null ? "idle" : ambientAffect;
        // The complete face sprites encode emotion in the mouth and cheeks as
        // well as in the eyes.  Circular presentation deliberately removes
        // those lower-face channels, so reuse FaceView's already synchronized
        // bilateral emotional closure to keep happy, affectionate, sad and
        // concerned states legible from the eyes alone.  Full-face mode still
        // receives only the ordinary blink and remains pixel-compatible.
        presentationClosureTarget = circularEyeOnly
                ? clamp01(Math.max(blink, emotionalClosure))
                : clamp01(blink);
        // Affection reads as soft, half-lowered lids over a happy face — the
        // living lids hold any partial position cleanly (owner 06.10.2026).
        if ("affectionate".equals(safeAffect)) {
            presentationClosureTarget = Math.max(presentationClosureTarget, AFFECTIONATE_LIDS);
        }
        // Error shares the surprised source plate, but must not become the
        // same visual state after the mouth is removed. A bounded binocular
        // tension differentiates it without introducing independent glitches.
        if (circularEyeOnly && "error".equals(safeAffect)) {
            presentationClosureTarget = Math.max(
                    presentationClosureTarget, 0.72f);
        }
        int semanticPose = speechAmount >= 0f ? HAPPY : resolvePose(safeActivity, safeAffect);
        int ambientPose = resolvePose("idle", safeAmbientAffect);
        // A slow alpha blend between two opaque raster faces creates two
        // mouths, brows and cheek contours.  Idle life therefore selects one
        // complete approved key pose once its envelope becomes readable; the
        // existing bilateral blink transition swaps the whole face cleanly.
        // This remains visual-only: FaceView's semantic state is still idle.
        boolean ambientPoseReadable = "idle".equals(safeActivity) &&
                "idle".equals(safeAffect) && ambientPose != NEUTRAL &&
                ambientBlend >= 0.045f;
        int requested = ambientPoseReadable ? ambientPose : semanticPose;
        boolean wantsSleepyClosure = "sleepy".equals(safeAffect);
        if (!initialized) {
            initialized = true;
            fromPose = requested;
            targetPose = requested;
            transitionStarted = now;
            transitionDurationMs = transitionDuration(safeActivity, safeAffect);
            transitionInitialClosure = 0f;
            transitionHoldClosed = wantsSleepyClosure;
            lastSequence = sequence;
        } else if (requested != targetPose ||
                wantsSleepyClosure != transitionHoldClosed) {
            float carriedClosure = Math.max(
                    Math.max(transitionClosureAt(now), blink),
                    lastPresentedClosure);
            fromPose = visiblePoseAt(now);
            targetPose = requested;
            transitionStarted = now;
            transitionDurationMs = transitionDuration(safeActivity, safeAffect);
            transitionInitialClosure = carriedClosure;
            transitionHoldClosed = wantsSleepyClosure;
            lastSequence = sequence;
        } else if (sequence != lastSequence) {
            // Re-delivering an event may change conversational timing without
            // changing the bitmap pose. Do not blink-cut the same face again.
            lastSequence = sequence;
        }

        // Keep progress linear here. Each close/open half applies smootherStep
        // exactly once; nesting the quintic packed most eyelid travel into only
        // a few middle frames and was the main source of visible jerkiness.
        float progress = clamp01((now - transitionStarted) / transitionDurationMs);
        if (progress < 1f) {
            drawTransition(canvas, fromPose, targetPose, progress, gazeX, gazeY);
        } else {
            float effectiveBlink = "sleepy".equals(safeAffect)
                    ? 1f : presentationClosureTarget;
            // Keep the original full-face settled path explicit: besides
            // avoiding an unnecessary presentation branch, this preserves the
            // audited v4.5 renderer contract byte-for-byte when the reversible
            // circular presentation is disabled.
            if (circularEyeOnly) {
                drawPresentedPose(canvas, targetPose, gazeX, gazeY,
                        effectiveBlink);
            } else {
                drawLayeredPose(canvas, targetPose, gazeX, gazeY,
                        effectiveBlink);
            }
            lastPresentedClosure = effectiveBlink;
        }
    }

    private static int resolvePose(String activity, String affect) {
        switch (affect) {
            case "happy": return HAPPY;
            case "laughing": return LAUGHING;
            case "curious": return CURIOUS;
            case "surprised": return SURPRISED;
            case "affectionate": return HAPPY;
            case "sleepy": return NEUTRAL;
            case "sad": return SAD;
            case "concerned": return CONCERNED;
            case "annoyed": return ANNOYED;
            case "error": return SURPRISED;
            default:
                break;
        }
        if ("thinking".equals(activity)) return CURIOUS;
        if ("speaking".equals(activity)) return HAPPY;
        return NEUTRAL;
    }

    private static float transitionDuration(String activity, String affect) {
        if ("surprised".equals(affect) || "listening".equals(activity)) return 270f;
        if ("sad".equals(affect) || "concerned".equals(affect) ||
                "sleepy".equals(affect)) return 480f;
        if ("happy".equals(affect) || "laughing".equals(affect) ||
                "affectionate".equals(affect)) return 380f;
        return 340f;
    }

    private int visiblePoseAt(long now) {
        float previousProgress = clamp01(
                (now - transitionStarted) / transitionDurationMs);
        if (transitionHoldClosed) {
            return previousProgress < 1f ? fromPose : targetPose;
        }
        return previousProgress < 0.5f ? fromPose : targetPose;
    }

    private float transitionClosureAt(long now) {
        float progress = clamp01(
                (now - transitionStarted) / transitionDurationMs);
        if (transitionHoldClosed) {
            return lerp(transitionInitialClosure, 1f, smootherStep(progress));
        }
        if (progress >= 1f) return 0f;
        if (progress < 0.5f) {
            return lerp(transitionInitialClosure, 1f,
                    smootherStep(progress * 2f));
        }
        return smootherStep((1f - progress) * 2f);
    }

    private void drawTransition(Canvas canvas, int from, int target,
                                float progress, float gazeX, float gazeY) {
        // Peak closure is the only instant at which the exact key pose changes.
        // A tiny whole-head squash reinforces the blink and hides the cut while
        // never cross-fading incompatible eyes, brows, mouths, or silhouettes.
        float closure;
        int displayedPose;
        if (transitionHoldClosed) {
            closure = lerp(transitionInitialClosure, 1f, smootherStep(progress));
            displayedPose = from;
        } else if (progress < 0.5f) {
            closure = lerp(transitionInitialClosure, 1f,
                    smootherStep(progress * 2f));
            displayedPose = from;
        } else {
            // Open onto the target eye-only squint rather than opening fully
            // for one frame and snapping back down. The source-pose exchange is
            // still hidden at peak closure and the whole motion remains C2.
            float targetClosure = circularEyeOnly
                    ? presentationClosureTarget : 0f;
            closure = lerp(targetClosure, 1f,
                    smootherStep((1f - progress) * 2f));
            displayedPose = target;
        }
        float squashY = 1f - 0.018f * closure;
        int saved = canvas.save();
        // A full head can squash subtly during a blink cut. The two physical
        // portholes cannot: scaling their presentation would turn the circles
        // into ellipses even though the eyelids themselves remain animated.
        if (!circularEyeOnly) {
            canvas.scale(1f, squashY, BASE_W * 0.5f, BASE_H * 0.57f);
        }
        // The complete eye globes are rendered first, then the approved face
        // plate occludes them with its fur, wet rims and expression-specific
        // lids.  Gaze therefore stays live beneath a blink without ever being
        // pasted over the face.
        drawPresentedPose(canvas, displayedPose, gazeX, gazeY, closure);
        lastPresentedClosure = closure;
        canvas.restoreToCount(saved);
    }

    private void drawPresentedPose(Canvas canvas, int pose, float gazeX,
                                   float gazeY, float blink) {
        if (!circularEyeOnly) {
            drawLayeredPose(canvas, pose, gazeX, gazeY, blink);
            return;
        }
        drawEyePorthole(canvas, pose, true, gazeX, gazeY, blink);
        drawEyePorthole(canvas, pose, false, gazeX, gazeY, blink);
    }

    /**
     * Show one side of the complete approved renderer through a true circle.
     * The face is not flattened into a new sprite: the existing moving optical
     * globe, pose-specific lids, blink patch and brow artwork all remain live
     * inside the transformed crop.
     */
    private void drawEyePorthole(Canvas canvas, int pose, boolean left,
                                 float gazeX, float gazeY, float blink) {
        float destinationX = left ? LEFT_PORTHOLE_X : RIGHT_PORTHOLE_X;
        float sourceX = left ? LEFT_SOURCE_X : RIGHT_SOURCE_X;
        float sourceY = PORTHOLE_SOURCE_Y[pose];
        float scale = PORTHOLE_SCALE[pose];
        portholeClip.rewind();
        portholeClip.addCircle(destinationX, PORTHOLE_Y, PORTHOLE_RADIUS,
                Path.Direction.CW);

        int saved = canvas.save();
        canvas.clipPath(portholeClip);
        canvas.translate(destinationX, PORTHOLE_Y);
        canvas.scale(scale, scale);
        canvas.translate(-sourceX, -sourceY);
        drawLayeredPose(canvas, pose, gazeX, gazeY, blink);
        canvas.restoreToCount(saved);

        canvas.drawCircle(destinationX, PORTHOLE_Y, PORTHOLE_RADIUS,
                portholeEdgePaint);
    }

    private void drawLayeredPose(Canvas canvas, int pose, float gazeX,
                                 float gazeY, float blink) {
        if (pose != LAUGHING) {
            drawConjugateGaze(canvas, pose, gazeX, gazeY, blink);
        }
        // The accepted foreground owns the stationary wet rim and socket.  A
        // closing lid then crosses that rim, exactly as a real lid does; its
        // slightly larger clip hides the open oval at peak closure.
        drawKeyframe(canvas, pose, 255);
        drawBlink(canvas, pose, blink);
    }

    private void drawKeyframe(Canvas canvas, int pose, int alpha) {
        Bitmap keyframe = keyframes[pose];
        if (keyframe == null) return;
        layerPaint.setAlpha(alpha);
        if (pose == HAPPY && speechAmount >= 0f && !circularEyeOnly) {
            int index = 0;
            float open = 0.03f + 0.97f * clamp01(speechAmount);
            for (int row = 0; row <= JAW_ROWS; row++) {
                float y = BASE_H * row / JAW_ROWS;
                for (int column = 0; column <= JAW_COLUMNS; column++) {
                    float x = BASE_W * column / JAW_COLUMNS;
                    float side = Math.min(1f, Math.abs(x - 540f) / 145f);
                    float top = 545f - 72f * side * side;
                    float height = 88f * (float)Math.sqrt(1f - side * side);
                    float offset = Math.min(height, Math.max(0f, y - top));
                    if (y > top + height)
                        offset *= Math.max(0f, (720f - y) / (720f - top - height));
                    jawMesh[index++] = x;
                    jawMesh[index++] = y - offset * (1f - open);
                }
            }
            canvas.drawBitmapMesh(keyframe, JAW_COLUMNS, JAW_ROWS, jawMesh, 0,
                    null, 0, layerPaint);
        } else {
            canvas.drawBitmap(keyframe, null, designRect, layerPaint);
        }
        layerPaint.setAlpha(255);
    }

    private void drawConjugateGaze(Canvas canvas, int pose, float gazeX,
                                   float gazeY, float blink) {
        // At essentially full closure the translated interior is entirely
        // hidden. Keeping it active even at peak closure prevents a one-frame
        // black flash if subpixel antialiasing leaves a tiny lid gap.
        if (pose == LAUGHING) return;
        float normalizedX = clamp(gazeX, -0.58f, 0.58f) / 0.58f;
        float normalizedY = clamp(gazeY, -0.46f, 0.46f) / 0.46f;
        float normalizedRadius = (float)Math.sqrt(
                normalizedX * normalizedX + normalizedY * normalizedY);
        if (normalizedRadius > 0.94f) {
            float safeScale = 0.94f / normalizedRadius;
            normalizedX *= safeScale;
            normalizedY *= safeScale;
        }
        // Direction is exactly shared; each asymmetric socket maps that
        // direction to its own physically available travel.
        drawIndependentEyeInterior(canvas, pose, true,
                normalizedX, normalizedY);
        drawIndependentEyeInterior(canvas, pose, false,
                normalizedX, normalizedY);
    }

    /** Draw one complete moving optical globe under the painted socket/lids. */
    private void drawIndependentEyeInterior(Canvas canvas, int pose,
                                             boolean left, float normalizedX,
                                             float normalizedY) {
        Bitmap texture = eyeTextures[pose][left ? 0 : 1];
        if (texture == null || LEFT_SOCKET_RX[pose] <= 0f) return;
        float socketX = left ? LEFT_SOCKET_X[pose] : RIGHT_SOCKET_X[pose];
        float socketY = left ? LEFT_SOCKET_Y[pose] : RIGHT_SOCKET_Y[pose];
        float socketRx = left ? LEFT_SOCKET_RX[pose] : RIGHT_SOCKET_RX[pose];
        float socketRy = left ? LEFT_SOCKET_RY[pose] : RIGHT_SOCKET_RY[pose];

        // The backing globe is deliberately a little larger than the optical
        // aperture.  When the textured globe rolls, newly revealed sclera is
        // therefore warm white rather than transparent/black.
        float globeX = socketX + normalizedX * 1.2f;
        float globeY = socketY + normalizedY * 0.9f;
        featureDestination.set(globeX - socketRx - 4f,
                globeY - socketRy - 4f, globeX + socketRx + 4f,
                globeY + socketRy + 4f);
        opticalGradientMatrix.setScale(socketRx * 1.55f,
                socketRy * 1.55f);
        opticalGradientMatrix.postTranslate(
                globeX - socketRx * 0.24f + normalizedX * 1.8f,
                globeY - socketRy * 0.25f + normalizedY * 1.4f);
        scleraGradient.setLocalMatrix(opticalGradientMatrix);
        canvas.drawOval(featureDestination, scleraPaint);

        // Roll the complete approved optical texture: sclera shading, iris,
        // pupil and both catchlights stay rigidly coupled.  Overscan prevents
        // the texture edge entering the socket; the alpha feather blends into
        // the backing globe at extreme gaze.
        float travelX = normalizedX * Math.min(14f, socketRx * 0.28f);
        float travelY = normalizedY * Math.min(9f, socketRy * 0.19f);
        float halfWidth = socketRx * 1.09f;
        float halfHeight = socketRy * 1.09f;
        featureDestination.set(socketX + travelX - halfWidth,
                socketY + travelY - halfHeight,
                socketX + travelX + halfWidth,
                socketY + travelY + halfHeight);
        canvas.drawBitmap(texture, null, featureDestination, layerPaint);
    }

    private void drawBlink(Canvas canvas, int pose, float blink) {
        if (pose == LAUGHING || blink < 0.015f || keyframes[pose] == null || lidMasks[pose][0] == null) return;
        float amount = clamp01(blink);
        drawLids(canvas, pose, true, amount);
        drawLids(canvas, pose, false, amount);
    }

    private float lidEdge(float top, float bottom, float cy, float ry, float u, float upper) {
        float open = (top + bottom) * 0.5f - (bottom - top) * 0.5f *
                (float)Math.sqrt(Math.max(0f, 1f - u * u)) - 3f;
        float closed = cy + ry * (0.18f + 0.10f * (1f - u * u));
        return lerp(open, closed, upper);
    }

    /** The lid's cover region: the aperture plus the painted rim, taller at the top. */
    private float lidTop(float cy, float ry) { return cy - ry - LID_TOP_RIM; }
    private float lidBottom(float cy, float ry) { return cy + ry + LID_RIM; }

    /**
     * Per pose and eye, once: a colour gain so the temple fur matches the ring
     * of fur around the eye, and a soft-edged alpha mask of the lid region
     * (Android 8.1 hardware canvases cannot blur a clip at draw time).
     */
    private void prepareLid(Bitmap source, int pose, boolean left) {
        int eye = left ? 0 : 1;
        float cx = left ? LEFT_SOCKET_X[pose] : RIGHT_SOCKET_X[pose];
        float cy = left ? LEFT_SOCKET_Y[pose] : RIGHT_SOCKET_Y[pose];
        float rx = (left ? LEFT_SOCKET_RX[pose] : RIGHT_SOCKET_RX[pose]) + LID_RIM;
        float ry = left ? LEFT_SOCKET_RY[pose] : RIGHT_SOCKET_RY[pose];
        float top = lidTop(cy, ry), bottom = lidBottom(cy, ry);
        float ocy = (top + bottom) * 0.5f, ory = (bottom - top) * 0.5f;
        double[] ring = new double[4], temple = new double[4];
        int x0 = Math.max(0, (int)(cx - rx - 40)), x1 = Math.min(source.getWidth() - 1, (int)(cx + rx + 40));
        int y0 = Math.max(0, (int)(top - 30)), y1 = Math.min(source.getHeight() - 1, (int)(bottom + 30));
        float side = left ? -1f : 1f;
        float tA = cx + side * (rx + 6f), tB = cx + side * (rx + 30f);
        float tl = Math.min(tA, tB), tr = Math.max(tA, tB);
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                int c = source.getPixel(x, y);
                float luma = (Color.red(c) + Color.green(c) + Color.blue(c)) / 3f;
                if (luma <= 60f || luma >= 190f) continue;
                float dx = (x + 0.5f - cx), dy = (y + 0.5f - ocy);
                float inner = (dx * dx) / ((rx + 4) * (rx + 4)) + (dy * dy) / ((ory + 4) * (ory + 4));
                float outer = (dx * dx) / ((rx + 16) * (rx + 16)) + (dy * dy) / ((ory + 16) * (ory + 16));
                if (inner > 1f && outer <= 1f) accumulate(ring, c);
                if (x >= tl && x <= tr && y >= cy - ry * 0.6f && y <= cy + ry * 0.6f) accumulate(temple, c);
            }
        }
        float[] gain = {1f, 1f, 1f};
        if (ring[3] > 0 && temple[3] > 0) {
            for (int k = 0; k < 3; k++) {
                gain[k] = clamp((float)(ring[k] / ring[3]) / Math.max(1f, (float)(temple[k] / temple[3])), 0.6f, 1.4f);
            }
        }
        lidFilters[pose][eye] = new ColorMatrixColorFilter(new float[]{
                gain[0], 0, 0, 0, 0,  0, gain[1], 0, 0, 0,  0, 0, gain[2], 0, 0,  0, 0, 0, 1, 0});
        int width = (int)Math.ceil(2 * rx + 16), height = (int)Math.ceil(bottom - top + 16);
        Bitmap mask = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8);
        int[] alpha = new int[width * height];
        float feather = LID_FEATHER;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float dx = (x + 0.5f - width * 0.5f) / (rx + 1f);
                float dy = (y + 0.5f - height * 0.5f) / (ory + 1f);
                float distance = (float)Math.sqrt(dx * dx + dy * dy);
                float edge = feather / Math.max(1f, Math.min(rx, ory));
                alpha[y * width + x] = Math.round(255 * clamp01((1f - distance) / edge + 0.5f)) << 24;
            }
        }
        mask.setPixels(alpha, 0, width, 0, 0, width, height);
        lidMasks[pose][eye] = mask;
    }

    private static void accumulate(double[] sum, int color) {
        sum[0] += Color.red(color); sum[1] += Color.green(color); sum[2] += Color.blue(color); sum[3] += 1;
    }

    /**
     * Eyelids grown from the pose's own fur (owner 06.10.2026).
     *
     * <p>The old blink pasted a scaled closed-eye crop from the laughing sheet
     * through a sliding mask: its painted lash stayed still while the mask
     * moved, the pose's own upper wet rim stayed visible above it (two lash
     * lines: "the lid slid off the eye"), and the foreign fur showed a round
     * seam. Now the upper lid is colour-matched fur from the same pose's outer
     * temple; it covers the painted rim with a soft edge, and one lash line
     * rides its moving edge down to a relaxed closed curve. A lower lid rises
     * late to meet it.</p>
     */
    private void drawLids(Canvas canvas, int pose, boolean left, float amount) {
        int eye = left ? 0 : 1;
        float cx = left ? LEFT_SOCKET_X[pose] : RIGHT_SOCKET_X[pose];
        float cy = left ? LEFT_SOCKET_Y[pose] : RIGHT_SOCKET_Y[pose];
        float rx = left ? LEFT_SOCKET_RX[pose] : RIGHT_SOCKET_RX[pose];
        float ry = left ? LEFT_SOCKET_RY[pose] : RIGHT_SOCKET_RY[pose];
        float orx = rx + LID_RIM;
        float top = lidTop(cy, ry), bottom = lidBottom(cy, ry);
        float upper = clamp01(amount / 0.85f);
        float lower = clamp01((amount - 0.45f) / 0.55f);

        upperLidClip.rewind();
        upperLidClip.moveTo(cx - orx - 4f, top - 8f);
        upperLidClip.lineTo(cx + orx + 4f, top - 8f);
        for (int i = LID_SAMPLES - 1; i >= 0; i--) {
            float u = -1.08f + 2.16f * i / (LID_SAMPLES - 1);
            upperLidClip.lineTo(cx + u * orx, lidEdge(top, bottom, cy, ry, u, upper));
        }
        upperLidClip.close();
        lidUnion.rewind();
        for (int i = 0; i < LID_SAMPLES; i++) {
            float u = -1.08f + 2.16f * i / (LID_SAMPLES - 1);
            float y = lidEdge(top, bottom, cy, ry, u, upper);
            if (i == 0) lidUnion.moveTo(cx + u * orx, y); else lidUnion.lineTo(cx + u * orx, y);
        }

        float side = left ? -1f : 1f;
        float templeA = cx + side * (orx + 6f), templeB = cx + side * (orx + 30f);
        templeSource.set(Math.round(Math.min(templeA, templeB)), Math.round(cy - ry * 0.6f),
                Math.round(Math.max(templeA, templeB)), Math.round(cy + ry * 0.6f));
        lidPaint.setColorFilter(lidFilters[pose][eye]);

        Bitmap mask = lidMasks[pose][eye];
        apertureDestination.set(cx - mask.getWidth() * 0.5f, (top + bottom) * 0.5f - mask.getHeight() * 0.5f,
                cx + mask.getWidth() * 0.5f, (top + bottom) * 0.5f + mask.getHeight() * 0.5f);
        int layer = canvas.saveLayer(apertureDestination, null);
        int saved = canvas.save();
        canvas.clipPath(upperLidClip);
        featureDestination.set(cx - orx - 6f, top - 8f, cx + orx + 6f, cy + ry * 0.6f);
        canvas.drawBitmap(keyframes[pose], templeSource, featureDestination, lidPaint);
        // The socket sits in shadow: darken the lid towards its rim so it sinks
        // into the face instead of reading as a flat disc.
        lidShadePaint.setShader(new RadialGradient(cx, (top + bottom) * 0.5f,
                Math.max(orx, (bottom - top) * 0.5f) * 1.05f,
                new int[]{Color.argb(0, 0, 0, 0), Color.argb(18, 30, 12, 0), Color.argb(70, 30, 12, 0)},
                new float[]{0f, 0.6f, 1f}, Shader.TileMode.CLAMP));
        canvas.drawRect(featureDestination, lidShadePaint);
        canvas.restoreToCount(saved);
        if (lower > 0f) {
            lowerLidClip.rewind();
            for (int i = 0; i < LID_SAMPLES; i++) {
                float u = -1.08f + 2.16f * i / (LID_SAMPLES - 1);
                float open = cy + (ry + LID_RIM) * (float)Math.sqrt(Math.max(0f, 1f - u * u)) + 3f;
                float closed = cy + ry * (0.30f + 0.05f * (1f - u * u));
                float y = lerp(open, closed, lower);
                if (i == 0) lowerLidClip.moveTo(cx + u * orx, y); else lowerLidClip.lineTo(cx + u * orx, y);
            }
            lowerLidClip.lineTo(cx + orx + 4f, bottom + 6f);
            lowerLidClip.lineTo(cx - orx - 4f, bottom + 6f);
            lowerLidClip.close();
            saved = canvas.save();
            canvas.clipPath(lowerLidClip);
            featureDestination.set(cx - orx - 6f, cy, cx + orx + 6f, bottom + 6f);
            canvas.drawBitmap(keyframes[pose], templeSource, featureDestination, lidPaint);
            canvas.restoreToCount(saved);
        }
        canvas.drawBitmap(mask, null, apertureDestination, lidMaskPaint);
        canvas.restoreToCount(layer);

        if (upper > 0.02f) {
            float width = 3.5f + 3.5f * upper;
            lashPaint.setStrokeWidth(width);
            saved = canvas.save();
            apertureDestination.set(cx - orx - 2f, top - 2f, cx + orx + 2f, bottom + 2f);
            portholeClip.rewind();
            portholeClip.addOval(apertureDestination, Path.Direction.CW);
            canvas.clipPath(portholeClip);
            canvas.translate(0f, width * 0.3f);
            canvas.drawPath(lidUnion, lashPaint);
            canvas.restoreToCount(saved);
        }
    }

    private static float smootherStep(float value) {
        value = clamp01(value);
        return value * value * value * (value * (value * 6f - 15f) + 10f);
    }

    private static float lerp(float from, float to, float amount) {
        return from + (to - from) * amount;
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static float clamp01(float value) {
        return clamp(value, 0f, 1f);
    }
}
