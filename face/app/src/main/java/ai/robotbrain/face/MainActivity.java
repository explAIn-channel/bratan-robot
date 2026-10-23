package ai.robotbrain.face;

import android.app.Activity;
import android.app.KeyguardManager;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

public final class MainActivity extends Activity {
    private FaceView faceView;
    private FaceServer faceServer;
    private MediaChannel mediaChannel;
    private CameraCapture cameraCapture;
    private AudioIO audioIO;
    private ProxyBroker proxyBroker;
    private BleMoodLight moodLight;
    private BleEars ears;                 // the body's microphones over Bluetooth (C52)
    private final Handler recoveryHandler = new Handler(Looper.getMainLooper());
    private long cameraUnreadySinceMs;
    private final Runnable privacyRefresh = new Runnable() {
        @Override public void run() {
            if (faceView != null) {
                faceView.setPrivacyActivity(
                        cameraCapture != null && cameraCapture.isCaptureActive(),
                        audioIO != null && audioIO.isCaptureActive(),
                        proxyBroker != null && proxyBroker.isNetworkActive());
                if (moodLight != null) {
                    moodLight.setSleeping(!faceView.displayAwake());
                    moodLight.show(faceView.getEmotionName());
                }
            }
            recoveryHandler.postDelayed(this, 250);
        }
    };
    private final Runnable cameraRecovery = new Runnable() {
        @Override public void run() {
            if (audioIO != null) audioIO.recoverCapture(
                checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED);
            if (cameraCapture != null &&
                    checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                if (cameraCapture.isReady()) {
                    cameraUnreadySinceMs = 0;
                } else {
                    long now = SystemClock.elapsedRealtime();
                    if (cameraUnreadySinceMs == 0) cameraUnreadySinceMs = now;
                    if (now - cameraUnreadySinceMs >= 4000) {
                        cameraCapture.restart();
                        cameraUnreadySinceMs = now;
                    }
                }
            }
            recoveryHandler.postDelayed(this, 3000);
        }
    };

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setStatusBarColor(Color.BLACK);
        getWindow().setNavigationBarColor(Color.BLACK);
        getWindow().addFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN |
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON |
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED |
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON |
                WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD);
        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.screenBrightness = 0.16f;
        getWindow().setAttributes(lp);
        if (android.os.Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
            KeyguardManager km = (KeyguardManager) getSystemService(KEYGUARD_SERVICE);
            if (km != null) km.requestDismissKeyguard(this, null);
        }

        faceView = new FaceView(this);
        setContentView(faceView);
        enterImmersive();

        cameraCapture = new CameraCapture(this, "1", 640, 480, 4);
        audioIO = new AudioIO(faceView);
        cameraCapture.setMicrophone(audioIO);
        moodLight = new BleMoodLight(this);
        ears = new BleEars(this);
        boolean cameraGranted = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        boolean audioGranted = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
        boolean lightGranted = checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED;
        if (cameraGranted) {
            cameraCapture.start();
        }
        if (audioGranted) {
            audioIO.start();
        }
        if (lightGranted) {
            moodLight.start();
            ears.start();
        }
        if (!cameraGranted || !audioGranted || !lightGranted) {
            requestPermissions(new String[]{Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.ACCESS_FINE_LOCATION}, 2701);
        }
        proxyBroker = new ProxyBroker(this);
        faceServer = new FaceServer(faceView, cameraCapture, audioIO, proxyBroker, moodLight);
        faceServer.start();
        mediaChannel = new MediaChannel(audioIO, cameraCapture, proxyBroker);
        mediaChannel.start();
        recoveryHandler.postDelayed(cameraRecovery, 3000);
        recoveryHandler.post(privacyRefresh);
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode != 2701) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                cameraCapture != null) {
            cameraCapture.start();
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                audioIO != null) {
            audioIO.start();
        }
        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED && moodLight != null) {
            moodLight.start();
            if (ears != null) ears.start();
        }
    }

    @Override protected void onResume() {
        super.onResume();
        enterImmersive();
    }

    @Override public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) enterImmersive();
    }

    private void enterImmersive() {
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                View.SYSTEM_UI_FLAG_FULLSCREEN |
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    @Override protected void onDestroy() {
        recoveryHandler.removeCallbacks(cameraRecovery);
        recoveryHandler.removeCallbacks(privacyRefresh);
        if (faceServer != null) faceServer.shutdown();
        if (mediaChannel != null) mediaChannel.shutdown();
        if (proxyBroker != null) proxyBroker.shutdown();
        if (cameraCapture != null) cameraCapture.shutdown();
        if (audioIO != null) audioIO.shutdown();
        if (moodLight != null) moodLight.shutdown();
        if (ears != null) ears.stop();
        super.onDestroy();
    }
}
