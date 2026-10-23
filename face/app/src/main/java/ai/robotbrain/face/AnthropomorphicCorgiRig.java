package ai.robotbrain.face;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;

/**
 * A compact 2D blendshape rig for the production corgi face.
 *
 * <p>The character remains a corgi, while expression grammar follows the same
 * independent facial actions used by human animation rigs: brows, upper lids,
 * lower lids/cheeks, nose, mouth corners, lips and jaw.  Activity (listening,
 * thinking, speaking), affect, binocular blink and conjugate gaze are composed
 * instead of replacing each other.</p>
 */
final class AnthropomorphicCorgiRig {
    private static final float BASE_W = 1080f;
    private static final float BASE_H = 720f;
    private static final float LEFT_EYE_X = 434f;
    private static final float RIGHT_EYE_X = BASE_W - LEFT_EYE_X;
    private static final float EYE_Y = 387f;

    private final RectF assetRect = new RectF(0f, 0f, BASE_W, BASE_H);
    private final RectF irisRect = new RectF();
    private final RectF tongueRect = new RectF();
    private final RectF cheekRect = new RectF();
    private final RectF leftLidRect = new RectF(365f, 306f, 505f, 465f);
    private final RectF rightLidRect = new RectF(575f, 306f, 715f, 465f);

    private final Paint bitmapPaint = new Paint(
            Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint globePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint socketPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pupilPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint reflectionPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint lidEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint wetEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint browPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint browHighlightPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint cheekPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint creasePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mouthFillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mouthEdgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint tonguePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Path eyePath = new Path();
    private final Path upperCoverPath = new Path();
    private final Path lowerCoverPath = new Path();
    private final Path upperBoundaryPath = new Path();
    private final Path lowerBoundaryPath = new Path();
    private final Path reflectionPath = new Path();
    private final Path mouthPath = new Path();
    private final Path lipPath = new Path();
    private final Path cheekPath = new Path();
    private final Path creasePath = new Path();
    private final Path browPath = new Path();

    private final Bitmap baseFace;
    private final Bitmap closedLeftEye;
    private final Bitmap closedRightEye;
    private final Bitmap irisTexture;

    private final Pose fromPose = new Pose();
    private final Pose currentPose = new Pose();
    private final Pose targetPose = new Pose();
    private final Pose affectPose = new Pose();
    private String lastActivity = "";
    private String lastAffect = "";
    private float lastIntensity = -1f;
    private long poseStarted;

    AnthropomorphicCorgiRig(Context context) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        baseFace = decodeScaled(context, R.drawable.corgi_cartoon_face_base_v2,
                (int)BASE_W, (int)BASE_H, options);
        Bitmap closedPlate = decodeScaled(context,
                R.drawable.corgi_cartoon_face_closed_v2,
                (int)BASE_W, (int)BASE_H, options);
        if (closedPlate != null) {
            closedLeftEye = cropScaled(closedPlate,
                    365, 306, 140, 159, 140, 159);
            closedRightEye = cropScaled(closedPlate,
                    575, 306, 140, 159, 140, 159);
            closedPlate.recycle();
        } else {
            closedLeftEye = null;
            closedRightEye = null;
        }
        irisTexture = decodeScaled(context, R.drawable.corgi_cartoon_iris_v2,
                192, 192, options);

        globePaint.setStyle(Paint.Style.FILL);
        globePaint.setColor(Color.rgb(255, 239, 207));
        socketPaint.setStyle(Paint.Style.STROKE);
        socketPaint.setStrokeCap(Paint.Cap.ROUND);
        socketPaint.setStrokeJoin(Paint.Join.ROUND);
        socketPaint.setStrokeWidth(6.5f);
        socketPaint.setColor(Color.rgb(68, 29, 14));
        pupilPaint.setStyle(Paint.Style.FILL);
        pupilPaint.setColor(Color.rgb(4, 3, 2));
        reflectionPaint.setStyle(Paint.Style.FILL);
        lidEdgePaint.setStyle(Paint.Style.STROKE);
        lidEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        lidEdgePaint.setStrokeJoin(Paint.Join.ROUND);
        lidEdgePaint.setStrokeWidth(2.2f);
        lidEdgePaint.setColor(Color.rgb(66, 31, 16));
        lidEdgePaint.setAlpha(125);
        wetEdgePaint.setStyle(Paint.Style.STROKE);
        wetEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        wetEdgePaint.setStrokeJoin(Paint.Join.ROUND);
        wetEdgePaint.setStrokeWidth(1.0f);
        wetEdgePaint.setColor(Color.rgb(235, 215, 193));
        wetEdgePaint.setAlpha(52);
        browPaint.setStyle(Paint.Style.FILL);
        browPaint.setColor(Color.rgb(116, 53, 19));
        browHighlightPaint.setStyle(Paint.Style.FILL);
        browHighlightPaint.setColor(Color.rgb(221, 112, 35));
        cheekPaint.setStyle(Paint.Style.FILL);
        creasePaint.setStyle(Paint.Style.STROKE);
        creasePaint.setStrokeCap(Paint.Cap.ROUND);
        creasePaint.setStrokeJoin(Paint.Join.ROUND);
        mouthFillPaint.setStyle(Paint.Style.FILL);
        mouthFillPaint.setColor(Color.rgb(48, 13, 13));
        mouthEdgePaint.setStyle(Paint.Style.STROKE);
        mouthEdgePaint.setStrokeCap(Paint.Cap.ROUND);
        mouthEdgePaint.setStrokeJoin(Paint.Join.ROUND);
        mouthEdgePaint.setColor(Color.rgb(69, 33, 23));
        tonguePaint.setStyle(Paint.Style.FILL);
        tonguePaint.setColor(Color.rgb(196, 75, 88));
    }

    private static Bitmap decodeScaled(Context context, int resource,
                                       int width, int height,
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

    private static Bitmap cropScaled(Bitmap source, int left, int top,
                                     int width, int height,
                                     int outputWidth, int outputHeight) {
        Bitmap output = Bitmap.createBitmap(
                outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        RectF destination = new RectF(0f, 0f, outputWidth, outputHeight);
        android.graphics.Rect crop = new android.graphics.Rect(
                left, top, left + width, top + height);
        canvas.drawBitmap(source, crop, destination, paint);
        return output;
    }

    void draw(Canvas canvas, String activity, String affect, float intensity,
              float sharedClosure, float gazeX, float gazeY, float irisScale,
              float pupilScale, long now, long activityStarted,
              long affectStarted, long sequence) {
        String safeActivity = activity == null ? "idle" : activity;
        String safeAffect = affect == null ? "idle" : affect;
        float safeIntensity = clamp01(intensity);
        buildTargetPose(targetPose, safeActivity, safeAffect, safeIntensity,
                now, activityStarted, affectStarted, sequence);

        boolean changed = !safeActivity.equals(lastActivity) ||
                !safeAffect.equals(lastAffect) ||
                Math.abs(lastIntensity - safeIntensity) > 0.005f;
        if (lastActivity.isEmpty()) {
            currentPose.copyFrom(targetPose);
            fromPose.copyFrom(targetPose);
            poseStarted = now;
        } else if (changed) {
            fromPose.copyFrom(currentPose);
            poseStarted = now;
        }
        lastActivity = safeActivity;
        lastAffect = safeAffect;
        lastIntensity = safeIntensity;

        float duration = transitionDuration(safeActivity, safeAffect);
        float progress = smootherStep((now - poseStarted) / duration);
        currentPose.interpolate(fromPose, targetPose, progress);

        float transientRoll = 0f;
        if ("laughing".equals(safeAffect) && now - affectStarted < 2600L) {
            float envelope = 1f - smoothStep((now - affectStarted - 1500f) / 900f);
            transientRoll = 0.75f * (float)Math.sin((now - affectStarted) / 88.0) *
                    clamp01(envelope);
        }

        int faceSave = canvas.save();
        float laughBounce = "laughing".equals(safeAffect) &&
                now - affectStarted < 2400L
                ? 3.5f * (float)Math.sin((now - affectStarted) / 105.0)
                : 0f;
        canvas.translate(0f, currentPose.headBob + laughBounce);
        canvas.scale(1.018f * (1f + currentPose.headScaleX),
                1.018f * (1f + currentPose.headScaleY),
                BASE_W * 0.5f, BASE_H * 0.54f);
        canvas.rotate(currentPose.headRoll + transientRoll,
                BASE_W * 0.5f, BASE_H * 0.53f);
        if (baseFace != null) {
            canvas.drawBitmap(baseFace, null, assetRect, bitmapPaint);
        }

        drawEye(canvas, true, gazeX, gazeY, irisScale, pupilScale,
                safeActivity, safeAffect, now, activityStarted, affectStarted);
        drawEye(canvas, false, gazeX, gazeY, irisScale, pupilScale,
                safeActivity, safeAffect, now, activityStarted, affectStarted);
        drawCheekMass(canvas, currentPose);
        drawBrows(canvas, currentPose);
        drawLids(canvas, true, currentPose, sharedClosure, gazeY);
        drawLids(canvas, false, currentPose, sharedClosure, gazeY);
        drawMuzzleAndMouth(canvas, currentPose, safeActivity, safeAffect,
                now, activityStarted, affectStarted);
        canvas.restoreToCount(faceSave);
    }

    private static float transitionDuration(String activity, String affect) {
        if ("surprised".equals(affect) || "listening".equals(activity)) return 210f;
        if ("sad".equals(affect) || "concerned".equals(affect) ||
                "sleepy".equals(affect)) return 360f;
        return 260f;
    }

    private void buildTargetPose(Pose out, String activity, String affect,
                                 float intensity, long now,
                                 long activityStarted, long affectStarted,
                                 long sequence) {
        out.reset();
        affectPose.reset();
        switch (affect) {
            case "happy":
                affectPose.cheekRaise = 0.88f;
                affectPose.cheekPuff = 0.34f;
                affectPose.lowerLidRaise = 0.64f;
                affectPose.browInnerUp = 0.18f;
                affectPose.browOuterUpL = affectPose.browOuterUpR = 0.14f;
                affectPose.smileL = affectPose.smileR = 0.96f;
                affectPose.lipsPart = 0.19f;
                affectPose.jawOpen = 0.10f;
                affectPose.headScaleX = 0.018f;
                affectPose.headScaleY = -0.010f;
                affectPose.headBob = -2f;
                break;
            case "laughing": {
                float elapsed = Math.max(0f, now - affectStarted);
                float pulse = 0.5f + 0.5f * (float)Math.sin(elapsed / 105.0);
                float envelope = 1f - smoothStep((elapsed - 1900f) / 900f);
                affectPose.cheekRaise = 1.0f;
                affectPose.cheekPuff = 0.48f;
                affectPose.lowerLidRaise = 0.90f;
                affectPose.browInnerUp = 0.14f;
                affectPose.browOuterUpL = affectPose.browOuterUpR = 0.16f;
                affectPose.smileL = affectPose.smileR = 0.96f;
                affectPose.lipsPart = 0.78f;
                affectPose.jawOpen = 0.66f + 0.22f * pulse * clamp01(envelope);
                affectPose.mouthStretch = 0.52f;
                affectPose.headScaleX = 0.035f;
                affectPose.headScaleY = -0.020f;
                affectPose.headBob = -4f;
                break;
            }
            case "curious":
                affectPose.browInnerUp = 0.30f;
                if ((sequence & 1L) == 0L) {
                    affectPose.browOuterUpL = 0.55f;
                    affectPose.browOuterUpR = 0.18f;
                    affectPose.smileL = 0.18f;
                    affectPose.smileR = 0.30f;
                    affectPose.headRoll = -3.2f;
                } else {
                    affectPose.browOuterUpL = 0.18f;
                    affectPose.browOuterUpR = 0.55f;
                    affectPose.smileL = 0.30f;
                    affectPose.smileR = 0.18f;
                    affectPose.headRoll = 3.2f;
                }
                affectPose.eyeWide = 0.18f;
                affectPose.pucker = 0.10f;
                affectPose.headScaleX = -0.006f;
                affectPose.headScaleY = 0.010f;
                break;
            case "surprised":
                affectPose.browInnerUp = 0.88f;
                affectPose.browOuterUpL = affectPose.browOuterUpR = 0.80f;
                affectPose.eyeWide = 0.84f;
                affectPose.lipsPart = 0.82f;
                affectPose.jawOpen = 0.58f;
                affectPose.pucker = 0.52f;
                affectPose.headScaleX = -0.018f;
                affectPose.headScaleY = 0.035f;
                affectPose.headBob = -5f;
                break;
            case "affectionate":
                affectPose.browInnerUp = 0.30f;
                affectPose.browOuterUpL = affectPose.browOuterUpR = 0.16f;
                affectPose.cheekRaise = 0.60f;
                affectPose.cheekPuff = 0.28f;
                affectPose.lowerLidRaise = 0.18f;
                affectPose.smileL = affectPose.smileR = 0.62f;
                affectPose.pucker = 0.05f;
                affectPose.headRoll = (sequence & 1L) == 0L ? -3.8f : 3.8f;
                affectPose.headScaleX = 0.016f;
                affectPose.headScaleY = -0.006f;
                break;
            case "sleepy":
                affectPose.lidDroop = 0.58f;
                affectPose.browOuterUpL = affectPose.browOuterUpR = 0.02f;
                affectPose.jawOpen = 0.06f;
                affectPose.lipsPart = 0.04f;
                affectPose.headScaleX = 0.020f;
                affectPose.headScaleY = -0.025f;
                affectPose.headBob = 5f;
                break;
            case "sad":
                affectPose.browInnerUp = 0.94f;
                affectPose.lidDroop = 0.30f;
                affectPose.lowerLidRaise = 0.16f;
                affectPose.frownL = affectPose.frownR = 0.96f;
                affectPose.chinRaise = 0.48f;
                affectPose.lipPress = 0.18f;
                affectPose.headRoll = (sequence & 1L) == 0L ? -2.4f : 2.4f;
                affectPose.headScaleX = -0.008f;
                affectPose.headScaleY = -0.020f;
                affectPose.headBob = 6f;
                break;
            case "concerned":
                affectPose.browInnerUp = 0.54f;
                affectPose.browDownL = affectPose.browDownR = 0.38f;
                affectPose.lowerLidRaise = 0.16f;
                affectPose.frownL = affectPose.frownR = 0.22f;
                affectPose.lipPress = 0.42f;
                affectPose.mouthStretch = 0.24f;
                affectPose.lipsPart = 0.08f;
                affectPose.headScaleX = -0.006f;
                affectPose.headScaleY = 0.010f;
                break;
            case "annoyed":
                affectPose.browDownL = affectPose.browDownR = 0.96f;
                affectPose.lidDroop = 0.16f;
                affectPose.lowerLidRaise = 0.72f;
                affectPose.lipPress = 0.86f;
                affectPose.dimpleL = 0.38f;
                affectPose.frownR = 0.38f;
                affectPose.noseSneer = 0.48f;
                affectPose.headScaleX = 0.025f;
                affectPose.headScaleY = -0.018f;
                affectPose.headBob = 2f;
                break;
            case "error": {
                float elapsed = Math.max(0f, now - affectStarted);
                float alarm = 0.35f + 0.65f *
                        (1f - smoothStep((elapsed - 230f) / 360f));
                affectPose.browInnerUp = lerp(0.62f, 0.94f, alarm);
                affectPose.browOuterUpL = affectPose.browOuterUpR =
                        lerp(0.34f, 0.88f, alarm);
                affectPose.browDownL = affectPose.browDownR = 0.22f * (1f - alarm);
                affectPose.eyeWide = 0.76f * alarm;
                affectPose.frownL = affectPose.frownR = 0.16f * (1f - alarm);
                affectPose.lipPress = 0.24f * (1f - alarm);
                affectPose.lipsPart = 0.38f * alarm;
                affectPose.jawOpen = 0.24f * alarm;
                affectPose.headScaleX = -0.012f * alarm;
                affectPose.headScaleY = 0.022f * alarm;
                break;
            }
            default:
                affectPose.browInnerUp = 0.025f;
                affectPose.smileL = affectPose.smileR = 0.025f;
                break;
        }

        float affectWeight = "idle".equals(affect) ? 1f : 0.62f + 0.38f * intensity;
        out.addScaled(affectPose, affectWeight);

        long activityElapsed = Math.max(0L, now - activityStarted);
        switch (activity) {
            case "listening": {
                float wake = 1f - smoothStep(activityElapsed / 700f);
                out.browInnerUp = Math.max(out.browInnerUp, 0.20f + 0.16f * wake);
                out.browOuterUpL = Math.max(out.browOuterUpL, 0.11f + 0.08f * wake);
                out.browOuterUpR = Math.max(out.browOuterUpR, 0.11f + 0.08f * wake);
                out.eyeWide = Math.max(out.eyeWide, 0.15f + 0.14f * wake);
                out.cheekRaise = Math.max(out.cheekRaise, 0.12f);
                out.lowerLidRaise = Math.max(out.lowerLidRaise, 0.07f);
                out.smileL = Math.max(out.smileL, 0.10f);
                out.smileR = Math.max(out.smileR, 0.10f);
                out.headScaleY = Math.max(out.headScaleY, 0.012f);
                out.headBob = Math.min(out.headBob, -2f);
                if (Math.abs(out.headRoll) < 0.5f) {
                    out.headRoll = (sequence & 1L) == 0L ? -1.8f : 1.8f;
                }
                break;
            }
            case "thinking":
                if ((sequence & 1L) == 0L) {
                    out.browOuterUpL = Math.max(out.browOuterUpL, 0.48f);
                    out.browDownR = Math.max(out.browDownR, 0.18f);
                    if (Math.abs(out.headRoll) < 0.5f) out.headRoll = -2.2f;
                } else {
                    out.browOuterUpR = Math.max(out.browOuterUpR, 0.48f);
                    out.browDownL = Math.max(out.browDownL, 0.18f);
                    if (Math.abs(out.headRoll) < 0.5f) out.headRoll = 2.2f;
                }
                out.lowerLidRaise = Math.max(out.lowerLidRaise, 0.13f);
                out.lipPress = Math.max(out.lipPress, 0.25f);
                out.pucker = Math.max(out.pucker, 0.10f);
                break;
            case "speaking": {
                // Held mouth shapes read as articulated speech instead of the
                // old metronomic sine-wave chewing motion.
                float visemeStep = (activityElapsed % 660L) / 110f;
                int viseme = (int)visemeStep;
                float visemeMix = smootherStep(visemeStep - viseme);
                int nextViseme = (viseme + 1) % 6;
                float open = lerp(visemeOpen(viseme), visemeOpen(nextViseme),
                        visemeMix);
                float stretch = lerp(visemeStretch(viseme),
                        visemeStretch(nextViseme), visemeMix);
                float pucker = lerp(visemePucker(viseme),
                        visemePucker(nextViseme), visemeMix);
                out.lipsPart = Math.max(out.lipsPart, 0.24f + 0.66f * open);
                out.jawOpen = Math.max(out.jawOpen, 0.08f + 0.46f * open);
                out.mouthStretch = Math.max(out.mouthStretch, 0.30f * stretch);
                out.pucker = Math.max(out.pucker, 0.24f * pucker);
                out.cheekRaise = Math.max(out.cheekRaise, 0.08f);
                break;
            }
            default:
                break;
        }

        // Anatomical couplings: a warm smile recruits cheek and lower lid;
        // annoyance can still tighten the lower lid without becoming a smile.
        float smile = 0.5f * (out.smileL + out.smileR);
        out.cheekRaise = Math.max(out.cheekRaise, smile * 0.64f);
        out.lowerLidRaise = Math.max(out.lowerLidRaise, smile * 0.36f);
        out.clamp();
    }

    private void drawEye(Canvas canvas, boolean left, float gazeX, float gazeY,
                         float irisScale, float pupilScale, String activity,
                         String affect, long now, long activityStarted,
                         long affectStarted) {
        buildAperture(left);
        canvas.drawPath(eyePath, globePaint);
        canvas.drawPath(eyePath, socketPaint);
        int clip = canvas.save();
        canvas.clipPath(eyePath);
        float centerX = left ? LEFT_EYE_X : RIGHT_EYE_X;
        float gestureX = 0f;
        float gestureY = 0f;
        if ("listening".equals(activity) && now - activityStarted < 700L) {
            gestureY -= 7f * (1f - smoothStep((now - activityStarted) / 700f));
        }
        if ("laughing".equals(affect) && now - affectStarted < 2600L) {
            gestureY += 3.4f * (float)Math.sin((now - affectStarted) / 88.0);
        } else if ("error".equals(affect) && now - affectStarted < 800L) {
            float envelope = 1f - smoothStep((now - affectStarted - 260f) / 420f);
            gestureX += 2.4f * (float)Math.sin((now - affectStarted) / 44.0) *
                    clamp01(envelope);
        }
        float convergence = 1.1f + 0.5f * (1f - Math.min(1f, Math.abs(gazeX)));
        float irisX = centerX + gazeX * 22f + gestureX +
                (left ? convergence : -convergence);
        float irisY = EYE_Y + gazeY * 20f + gestureY;
        float radius = 42f * clamp(irisScale, 0.94f, 1.08f);
        irisRect.set(irisX - radius, irisY - radius, irisX + radius, irisY + radius);
        if (irisTexture != null) {
            canvas.drawBitmap(irisTexture, null, irisRect, bitmapPaint);
        } else {
            Paint fallback = globePaint;
            int old = fallback.getColor();
            fallback.setColor(Color.rgb(145, 73, 22));
            canvas.drawCircle(irisX, irisY, radius, fallback);
            fallback.setColor(old);
        }
        float pupil = radius * clamp(pupilScale, 0.38f, 0.78f);
        canvas.drawCircle(irisX, irisY, pupil, pupilPaint);

        reflectionPaint.setColor(Color.rgb(255, 248, 226));
        reflectionPaint.setAlpha(225);
        canvas.drawCircle(irisX - radius * 0.28f,
                irisY - radius * 0.31f, radius * 0.17f, reflectionPaint);
        reflectionPaint.setAlpha(120);
        canvas.drawCircle(irisX + radius * 0.09f,
                irisY - radius * 0.11f, radius * 0.055f, reflectionPaint);
        reflectionPaint.setAlpha(255);
        canvas.restoreToCount(clip);
    }

    private void drawCheekMass(Canvas canvas, Pose pose) {
        float lift = pose.cheekRaise;
        float puff = pose.cheekPuff;
        if (lift < 0.02f && puff < 0.02f) return;
        int alpha = Math.round(22f + 48f * Math.max(lift, puff));
        cheekPaint.setColor(Color.argb(alpha, 255, 235, 203));
        int cheekClip = canvas.save();
        buildAperture(true);
        canvas.clipOutPath(eyePath);
        buildAperture(false);
        canvas.clipOutPath(eyePath);
        canvas.clipOutRect(485f, 405f, 595f, 580f);
        drawCheek(canvas, true, lift, puff);
        drawCheek(canvas, false, lift, puff);
        canvas.restoreToCount(cheekClip);

        creasePaint.setColor(Color.argb(Math.round(20f + 55f * lift), 99, 47, 25));
        creasePaint.setStrokeWidth(1.6f + 1.6f * lift);
        drawSmileCrease(canvas, true, pose, 0.72f);
        drawSmileCrease(canvas, false, pose, 0.72f);
    }

    private void drawCheek(Canvas canvas, boolean left, float lift, float puff) {
        float centerX = x(left, 397f) + (left ? -1f : 1f) * puff * 8f;
        float centerY = 474f - lift * 22f;
        float rx = 92f + puff * 18f;
        float ry = 61f + puff * 12f;
        cheekPath.rewind();
        cheekRect.set(centerX - rx, centerY - ry, centerX + rx, centerY + ry);
        cheekPath.addOval(cheekRect, Path.Direction.CW);
        canvas.drawPath(cheekPath, cheekPaint);
    }

    private void drawBrows(Canvas canvas, Pose pose) {
        drawBrow(canvas, true, pose);
        drawBrow(canvas, false, pose);
    }

    private void drawBrow(Canvas canvas, boolean left, Pose pose) {
        float outerX = x(left, 367f);
        float innerX = x(left, 503f);
        float outerUp = left ? pose.browOuterUpL : pose.browOuterUpR;
        float down = left ? pose.browDownL : pose.browDownR;
        float outerY = 300f - outerUp * 52f + down * 21f;
        float innerY = 294f - pose.browInnerUp * 65f + down * 42f;
        innerX += (left ? 1f : -1f) * down * 12f;
        float arch = 13f + 17f * Math.max(outerUp, pose.browInnerUp) - 6f * down;
        float thickness = 14f + 8f * Math.max(
                Math.max(outerUp, pose.browInnerUp), down);
        float c1X = lerp(outerX, innerX, 0.34f);
        float c2X = lerp(outerX, innerX, 0.72f);

        // A broad, filled supra-orbital fur mass reads as part of the cartoon
        // forehead.  The former high-detail bitmap strip looked like a human
        // eyebrow glued onto realistic fur and was a major uncanny-valley cue.
        browPath.rewind();
        browPath.moveTo(outerX, outerY + thickness * 0.12f);
        browPath.cubicTo(c1X, outerY - arch - thickness * 0.48f,
                c2X, innerY - arch - thickness * 0.30f,
                innerX, innerY + thickness * 0.05f);
        browPath.cubicTo(c2X, innerY + thickness * 0.68f,
                c1X, outerY + thickness * 0.72f,
                outerX, outerY + thickness * 0.12f);
        browPath.close();
        browPaint.setAlpha(Math.round(205f + 35f * Math.max(down,
                Math.max(outerUp, pose.browInnerUp))));
        canvas.drawPath(browPath, browPaint);

        // A smaller warm fur plane embeds the expression in the illustrated
        // forehead instead of reading as a flat human eyebrow sticker.
        float inset = thickness * 0.30f;
        browPath.rewind();
        browPath.moveTo(lerp(outerX, innerX, 0.12f), outerY - inset * 0.10f);
        browPath.cubicTo(c1X, outerY - arch - inset * 0.28f,
                c2X, innerY - arch - inset * 0.12f,
                lerp(outerX, innerX, 0.88f), innerY + inset * 0.04f);
        browPath.cubicTo(c2X, innerY + inset * 0.30f,
                c1X, outerY + inset * 0.26f,
                lerp(outerX, innerX, 0.12f), outerY - inset * 0.10f);
        browPath.close();
        browHighlightPaint.setAlpha(Math.round(62f + 36f * Math.max(
                Math.max(outerUp, pose.browInnerUp), down)));
        canvas.drawPath(browPath, browHighlightPaint);
    }

    private void drawLids(Canvas canvas, boolean left, Pose pose,
                          float sharedClosure, float gazeY) {
        Bitmap lidPlate = left ? closedLeftEye : closedRightEye;
        if (lidPlate == null) return;
        float upper = clamp01(pose.lidDroop - pose.eyeWide * 0.14f);
        float lower = clamp01(pose.lowerLidRaise + pose.cheekRaise * 0.16f -
                pose.eyeWide * 0.08f);
        if (gazeY > 0f) {
            upper += gazeY * 0.045f;
            lower -= gazeY * 0.025f;
        } else {
            upper += gazeY * 0.035f;
            lower -= gazeY * 0.030f;
        }
        float fullBlinkMix = smoothStep((sharedClosure - 0.72f) / 0.28f);
        float lowerClosure = sharedClosure * lerp(0.44f, 1f, fullBlinkMix);
        upper = clamp01(upper + (1f - upper) * sharedClosure);
        lower = clamp01(lower + (1f - lower) * lowerClosure);
        if (upper < 0.006f && lower < 0.006f) return;

        buildAperture(left);
        int eyeClip = canvas.save();
        canvas.clipPath(eyePath);
        if (sharedClosure > 0.94f) {
            drawClosedEyePlate(canvas, left, lidPlate);
            canvas.restoreToCount(eyeClip);
            return;
        }

        float fullMix = fullBlinkMix;
        float canonicalSlope = (pose.browDownL + pose.browDownR) * 0.055f -
                pose.browInnerUp * 0.045f;
        float upperSlope = (left ? canonicalSlope : -canonicalSlope) * (1f - fullMix);
        float lowerSlope = (left ? -canonicalSlope : canonicalSlope) *
                0.45f * (1f - fullMix);
        buildUpperCover(left, upper, upperSlope);
        buildLowerCover(left, lower, lowerSlope);

        int upperClip = canvas.save();
        canvas.clipPath(upperCoverPath);
        drawClosedEyePlate(canvas, left, lidPlate);
        canvas.restoreToCount(upperClip);
        int lowerClip = canvas.save();
        canvas.clipPath(lowerCoverPath);
        drawClosedEyePlate(canvas, left, lidPlate);
        canvas.restoreToCount(lowerClip);

        lidEdgePaint.setAlpha(185);
        canvas.drawPath(upperBoundaryPath, lidEdgePaint);
        canvas.drawPath(lowerBoundaryPath, lidEdgePaint);
        if (lower > 0.09f && sharedClosure < 0.92f) {
            canvas.drawPath(lowerBoundaryPath, wetEdgePaint);
        }
        canvas.restoreToCount(eyeClip);
    }

    private void drawClosedEyePlate(Canvas canvas, boolean left, Bitmap lidPlate) {
        canvas.drawBitmap(lidPlate, null, left ? leftLidRect : rightLidRect,
                bitmapPaint);
    }

    private void drawMuzzleAndMouth(Canvas canvas, Pose pose, String activity,
                                    String affect, long now,
                                    long activityStarted, long affectStarted) {
        float smileAverage = 0.5f * (pose.smileL + pose.smileR);
        float frownAverage = 0.5f * (pose.frownL + pose.frownR);
        float centerX = BASE_W * 0.5f;
        float centerY = 578f - pose.chinRaise * 5f;
        float width = 112f + 47f * smileAverage + 34f * pose.mouthStretch -
                52f * pose.pucker;
        width = clamp(width, 63f, 188f);
        float leftX = centerX - width;
        float rightX = centerX + width;
        float leftY = centerY - pose.smileL * 36f + pose.frownL * 33f +
                pose.dimpleL * 5f;
        float rightY = centerY - pose.smileR * 36f + pose.frownR * 33f +
                pose.dimpleR * 5f;
        float upperY = centerY - 4f - pose.upperLipUp * 15f;
        float openness = clamp01(pose.lipsPart * 0.56f + pose.jawOpen * 0.72f);
        float lowerY = upperY + 10f + openness * (32f + 105f * pose.jawOpen) -
                pose.chinRaise * 8f;

        // Philtrum anchors the human-readable mouth to the existing dog nose.
        creasePaint.setColor(Color.argb(165, 63, 31, 23));
        creasePaint.setStrokeWidth(3.3f + pose.lipPress * 1.5f);
        creasePath.rewind();
        creasePath.moveTo(centerX, 548f);
        creasePath.cubicTo(centerX - 1.5f, 556f,
                centerX + 1.5f, 567f, centerX, upperY + 2f);
        canvas.drawPath(creasePath, creasePaint);

        if (openness > 0.08f) {
            mouthPath.rewind();
            mouthPath.moveTo(leftX, leftY);
            mouthPath.cubicTo(leftX + width * 0.28f, upperY - 7f,
                    centerX - width * 0.22f, upperY - 5f, centerX, upperY);
            mouthPath.cubicTo(centerX + width * 0.22f, upperY - 5f,
                    rightX - width * 0.28f, upperY - 7f, rightX, rightY);
            float bottomControl = lowerY + 15f * (1f - pose.pucker);
            mouthPath.cubicTo(rightX - width * 0.22f, bottomControl,
                    centerX + width * 0.20f, bottomControl, centerX, lowerY);
            mouthPath.cubicTo(centerX - width * 0.20f, bottomControl,
                    leftX + width * 0.22f, bottomControl, leftX, leftY);
            mouthPath.close();
            canvas.drawPath(mouthPath, mouthFillPaint);

            int mouthClip = canvas.save();
            canvas.clipPath(mouthPath);
            if (pose.jawOpen > 0.26f || "laughing".equals(affect)) {
                float tongueWidth = width * 0.88f;
                float tongueTop = lowerY - 11f - 16f * pose.jawOpen;
                tongueRect.set(centerX - tongueWidth * 0.5f, tongueTop,
                        centerX + tongueWidth * 0.5f,
                        lowerY + 25f + 12f * pose.jawOpen);
                tonguePaint.setAlpha(Math.round(145f + 90f * pose.jawOpen));
                canvas.drawOval(tongueRect, tonguePaint);
            }
            canvas.restoreToCount(mouthClip);
            mouthEdgePaint.setStrokeWidth(5.0f + pose.lipPress * 3f);
            mouthEdgePaint.setAlpha(232);
            canvas.drawPath(mouthPath, mouthEdgePaint);
        } else {
            mouthEdgePaint.setStrokeWidth(5.2f + pose.lipPress * 4.2f);
            mouthEdgePaint.setAlpha(Math.round(205f + 38f * pose.lipPress));
            lipPath.rewind();
            lipPath.moveTo(centerX, upperY + 2f);
            lipPath.cubicTo(centerX - width * 0.18f, upperY + 9f,
                    leftX + width * 0.34f, leftY + 5f, leftX, leftY);
            canvas.drawPath(lipPath, mouthEdgePaint);
            lipPath.rewind();
            lipPath.moveTo(centerX, upperY + 2f);
            lipPath.cubicTo(centerX + width * 0.18f, upperY + 9f,
                    rightX - width * 0.34f, rightY + 5f, rightX, rightY);
            canvas.drawPath(lipPath, mouthEdgePaint);
        }

        if (smileAverage > 0.12f || frownAverage > 0.20f) {
            creasePaint.setColor(Color.argb(
                    Math.round(32f + 58f * Math.max(smileAverage, frownAverage)),
                    88, 45, 28));
            creasePaint.setStrokeWidth(1.5f + 1.6f * smileAverage);
            drawSmileCrease(canvas, true, pose, 1f);
            drawSmileCrease(canvas, false, pose, 1f);
        }
        if (pose.noseSneer > 0.05f) {
            creasePaint.setColor(Color.argb(Math.round(70f + 130f * pose.noseSneer),
                    48, 22, 17));
            creasePaint.setStrokeWidth(2.4f + 3.2f * pose.noseSneer);
            creasePath.rewind();
            creasePath.moveTo(490f, 544f);
            creasePath.cubicTo(474f, 552f - 11f * pose.noseSneer,
                    468f, 568f, 457f, 577f);
            canvas.drawPath(creasePath, creasePaint);
            creasePath.rewind();
            creasePath.moveTo(590f, 544f);
            creasePath.cubicTo(606f, 552f - 11f * pose.noseSneer,
                    612f, 568f, 623f, 577f);
            canvas.drawPath(creasePath, creasePaint);
        }
    }

    private void drawSmileCrease(Canvas canvas, boolean left, Pose pose, float alphaScale) {
        float smile = left ? pose.smileL : pose.smileR;
        float frown = left ? pose.frownL : pose.frownR;
        float centerX = x(left, 422f);
        float startY = 548f - smile * 14f + frown * 7f;
        int originalAlpha = creasePaint.getAlpha();
        creasePaint.setAlpha(Math.round(originalAlpha * alphaScale));
        creasePath.rewind();
        creasePath.moveTo(centerX, startY);
        creasePath.cubicTo(x(left, 407f), 529f - pose.cheekRaise * 13f,
                x(left, 394f), 516f - pose.cheekRaise * 18f,
                x(left, 388f), 501f - pose.cheekRaise * 15f);
        canvas.drawPath(creasePath, creasePaint);
        creasePaint.setAlpha(originalAlpha);
    }

    private void buildAperture(boolean left) {
        float expansion = currentPose.eyeWide * 7f;
        float outerX = x(left, 386f - expansion);
        float innerX = x(left, 482f + expansion);
        float topY = 328f - currentPose.eyeWide * 24f;
        float bottomY = 446f + currentPose.eyeWide * 17f;
        eyePath.rewind();
        eyePath.moveTo(outerX, 386f);
        eyePath.cubicTo(x(left, 394f - expansion * 0.4f), topY,
                x(left, 471f + expansion * 0.4f), topY, innerX, 386f);
        eyePath.cubicTo(x(left, 471f + expansion * 0.4f), bottomY,
                x(left, 394f - expansion * 0.4f), bottomY, outerX, 386f);
        eyePath.close();
    }

    private void buildUpperCover(boolean left, float progress, float slope) {
        float expansion = currentPose.eyeWide * 7f;
        float outerX = x(left, 386f - expansion);
        float innerX = x(left, 482f + expansion);
        float topY = 328f - currentPose.eyeWide * 24f;
        float c1X = x(left, lerp(394f - expansion * 0.4f, 417f, progress));
        float c2X = x(left, lerp(471f + expansion * 0.4f, 452f, progress));
        float finalY = 387f;
        float outerY = lerp(386f, tiltedY(left, outerX, finalY, slope), progress);
        float c1Y = lerp(topY, tiltedY(left, c1X, finalY + 2f, slope), progress);
        float c2Y = lerp(topY, tiltedY(left, c2X, finalY + 1f, slope), progress);
        float innerY = lerp(386f, tiltedY(left, innerX, finalY, slope), progress);
        upperBoundaryPath.rewind();
        upperBoundaryPath.moveTo(outerX, outerY);
        upperBoundaryPath.cubicTo(c1X, c1Y, c2X, c2Y, innerX, innerY);
        upperCoverPath.rewind();
        upperCoverPath.moveTo(outerX, 386f);
        upperCoverPath.cubicTo(x(left, 394f - expansion * 0.4f), topY,
                x(left, 471f + expansion * 0.4f), topY, innerX, 386f);
        upperCoverPath.lineTo(innerX, innerY);
        upperCoverPath.cubicTo(c2X, c2Y, c1X, c1Y, outerX, outerY);
        upperCoverPath.close();
    }

    private void buildLowerCover(boolean left, float progress, float slope) {
        float expansion = currentPose.eyeWide * 7f;
        float outerX = x(left, 386f - expansion);
        float innerX = x(left, 482f + expansion);
        float bottomY = 446f + currentPose.eyeWide * 17f;
        float c1X = x(left, lerp(394f - expansion * 0.4f, 417f, progress));
        float c2X = x(left, lerp(471f + expansion * 0.4f, 452f, progress));
        float finalY = 387f;
        float outerY = lerp(386f, tiltedY(left, outerX, finalY, slope), progress);
        float c1Y = lerp(bottomY, tiltedY(left, c1X, finalY + 2f, slope), progress);
        float c2Y = lerp(bottomY, tiltedY(left, c2X, finalY + 1f, slope), progress);
        float innerY = lerp(386f, tiltedY(left, innerX, finalY, slope), progress);
        lowerBoundaryPath.rewind();
        lowerBoundaryPath.moveTo(outerX, outerY);
        lowerBoundaryPath.cubicTo(c1X, c1Y, c2X, c2Y, innerX, innerY);
        lowerCoverPath.rewind();
        lowerCoverPath.moveTo(innerX, 386f);
        lowerCoverPath.cubicTo(x(left, 471f + expansion * 0.4f), bottomY,
                x(left, 394f - expansion * 0.4f), bottomY, outerX, 386f);
        lowerCoverPath.lineTo(outerX, outerY);
        lowerCoverPath.cubicTo(c1X, c1Y, c2X, c2Y, innerX, innerY);
        lowerCoverPath.close();
    }

    private static float tiltedY(boolean left, float screenX, float baseY, float slope) {
        float center = left ? LEFT_EYE_X : RIGHT_EYE_X;
        return baseY + (screenX - center) * slope;
    }

    private static float visemeOpen(int viseme) {
        switch (viseme) {
            case 0: return 0.18f;
            case 1: return 0.72f;
            case 2: return 0.42f;
            case 3: return 0.88f;
            case 4: return 0.30f;
            default: return 0.56f;
        }
    }

    private static float visemeStretch(int viseme) {
        switch (viseme) {
            case 0: return 0.18f;
            case 1: return 0.48f;
            case 2: return 0.92f;
            case 3: return 0.32f;
            case 4: return 0.70f;
            default: return 0.20f;
        }
    }

    private static float visemePucker(int viseme) {
        switch (viseme) {
            case 0: return 0.18f;
            case 1: return 0.08f;
            case 2: return 0.04f;
            case 3: return 0.26f;
            case 4: return 0.08f;
            default: return 0.90f;
        }
    }

    private static float x(boolean left, float leftX) {
        return left ? leftX : BASE_W - leftX;
    }

    private static float smoothStep(float value) {
        value = clamp01(value);
        return value * value * (3f - 2f * value);
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

    private static final class Pose {
        float browInnerUp;
        float browOuterUpL;
        float browOuterUpR;
        float browDownL;
        float browDownR;
        float eyeWide;
        float lidDroop;
        float lowerLidRaise;
        float cheekRaise;
        float cheekPuff;
        float noseSneer;
        float smileL;
        float smileR;
        float frownL;
        float frownR;
        float dimpleL;
        float dimpleR;
        float upperLipUp;
        float lipPress;
        float pucker;
        float mouthStretch;
        float lipsPart;
        float jawOpen;
        float chinRaise;
        float headRoll;
        float headScaleX;
        float headScaleY;
        float headBob;

        void reset() {
            browInnerUp = 0f;
            browOuterUpL = browOuterUpR = 0f;
            browDownL = browDownR = 0f;
            eyeWide = lidDroop = lowerLidRaise = 0f;
            cheekRaise = cheekPuff = noseSneer = 0f;
            smileL = smileR = frownL = frownR = 0f;
            dimpleL = dimpleR = upperLipUp = 0f;
            lipPress = pucker = mouthStretch = 0f;
            lipsPart = jawOpen = chinRaise = 0f;
            headRoll = 0f;
            headScaleX = headScaleY = headBob = 0f;
        }

        void copyFrom(Pose other) {
            browInnerUp = other.browInnerUp;
            browOuterUpL = other.browOuterUpL;
            browOuterUpR = other.browOuterUpR;
            browDownL = other.browDownL;
            browDownR = other.browDownR;
            eyeWide = other.eyeWide;
            lidDroop = other.lidDroop;
            lowerLidRaise = other.lowerLidRaise;
            cheekRaise = other.cheekRaise;
            cheekPuff = other.cheekPuff;
            noseSneer = other.noseSneer;
            smileL = other.smileL;
            smileR = other.smileR;
            frownL = other.frownL;
            frownR = other.frownR;
            dimpleL = other.dimpleL;
            dimpleR = other.dimpleR;
            upperLipUp = other.upperLipUp;
            lipPress = other.lipPress;
            pucker = other.pucker;
            mouthStretch = other.mouthStretch;
            lipsPart = other.lipsPart;
            jawOpen = other.jawOpen;
            chinRaise = other.chinRaise;
            headRoll = other.headRoll;
            headScaleX = other.headScaleX;
            headScaleY = other.headScaleY;
            headBob = other.headBob;
        }

        void addScaled(Pose other, float scale) {
            browInnerUp += other.browInnerUp * scale;
            browOuterUpL += other.browOuterUpL * scale;
            browOuterUpR += other.browOuterUpR * scale;
            browDownL += other.browDownL * scale;
            browDownR += other.browDownR * scale;
            eyeWide += other.eyeWide * scale;
            lidDroop += other.lidDroop * scale;
            lowerLidRaise += other.lowerLidRaise * scale;
            cheekRaise += other.cheekRaise * scale;
            cheekPuff += other.cheekPuff * scale;
            noseSneer += other.noseSneer * scale;
            smileL += other.smileL * scale;
            smileR += other.smileR * scale;
            frownL += other.frownL * scale;
            frownR += other.frownR * scale;
            dimpleL += other.dimpleL * scale;
            dimpleR += other.dimpleR * scale;
            upperLipUp += other.upperLipUp * scale;
            lipPress += other.lipPress * scale;
            pucker += other.pucker * scale;
            mouthStretch += other.mouthStretch * scale;
            lipsPart += other.lipsPart * scale;
            jawOpen += other.jawOpen * scale;
            chinRaise += other.chinRaise * scale;
            headRoll += other.headRoll * scale;
            headScaleX += other.headScaleX * scale;
            headScaleY += other.headScaleY * scale;
            headBob += other.headBob * scale;
        }

        void interpolate(Pose from, Pose to, float amount) {
            browInnerUp = lerp(from.browInnerUp, to.browInnerUp, amount);
            browOuterUpL = lerp(from.browOuterUpL, to.browOuterUpL, amount);
            browOuterUpR = lerp(from.browOuterUpR, to.browOuterUpR, amount);
            browDownL = lerp(from.browDownL, to.browDownL, amount);
            browDownR = lerp(from.browDownR, to.browDownR, amount);
            eyeWide = lerp(from.eyeWide, to.eyeWide, amount);
            lidDroop = lerp(from.lidDroop, to.lidDroop, amount);
            lowerLidRaise = lerp(from.lowerLidRaise, to.lowerLidRaise, amount);
            cheekRaise = lerp(from.cheekRaise, to.cheekRaise, amount);
            cheekPuff = lerp(from.cheekPuff, to.cheekPuff, amount);
            noseSneer = lerp(from.noseSneer, to.noseSneer, amount);
            smileL = lerp(from.smileL, to.smileL, amount);
            smileR = lerp(from.smileR, to.smileR, amount);
            frownL = lerp(from.frownL, to.frownL, amount);
            frownR = lerp(from.frownR, to.frownR, amount);
            dimpleL = lerp(from.dimpleL, to.dimpleL, amount);
            dimpleR = lerp(from.dimpleR, to.dimpleR, amount);
            upperLipUp = lerp(from.upperLipUp, to.upperLipUp, amount);
            lipPress = lerp(from.lipPress, to.lipPress, amount);
            pucker = lerp(from.pucker, to.pucker, amount);
            mouthStretch = lerp(from.mouthStretch, to.mouthStretch, amount);
            lipsPart = lerp(from.lipsPart, to.lipsPart, amount);
            jawOpen = lerp(from.jawOpen, to.jawOpen, amount);
            chinRaise = lerp(from.chinRaise, to.chinRaise, amount);
            headRoll = lerp(from.headRoll, to.headRoll, amount);
            headScaleX = lerp(from.headScaleX, to.headScaleX, amount);
            headScaleY = lerp(from.headScaleY, to.headScaleY, amount);
            headBob = lerp(from.headBob, to.headBob, amount);
        }

        void clamp() {
            browInnerUp = clamp01(browInnerUp);
            browOuterUpL = clamp01(browOuterUpL);
            browOuterUpR = clamp01(browOuterUpR);
            browDownL = clamp01(browDownL);
            browDownR = clamp01(browDownR);
            eyeWide = clamp01(eyeWide);
            lidDroop = clamp01(lidDroop);
            lowerLidRaise = clamp01(lowerLidRaise);
            cheekRaise = clamp01(cheekRaise);
            cheekPuff = clamp01(cheekPuff);
            noseSneer = clamp01(noseSneer);
            smileL = clamp01(smileL);
            smileR = clamp01(smileR);
            frownL = clamp01(frownL);
            frownR = clamp01(frownR);
            dimpleL = clamp01(dimpleL);
            dimpleR = clamp01(dimpleR);
            upperLipUp = clamp01(upperLipUp);
            lipPress = clamp01(lipPress);
            pucker = clamp01(pucker);
            mouthStretch = clamp01(mouthStretch);
            lipsPart = clamp01(lipsPart);
            jawOpen = clamp01(jawOpen);
            chinRaise = clamp01(chinRaise);
            headRoll = AnthropomorphicCorgiRig.clamp(headRoll, -5f, 5f);
            headScaleX = AnthropomorphicCorgiRig.clamp(headScaleX, -0.04f, 0.05f);
            headScaleY = AnthropomorphicCorgiRig.clamp(headScaleY, -0.04f, 0.05f);
            headBob = AnthropomorphicCorgiRig.clamp(headBob, -10f, 10f);
        }
    }
}
