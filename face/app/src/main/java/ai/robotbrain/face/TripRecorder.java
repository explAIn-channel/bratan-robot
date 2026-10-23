package ai.robotbrain.face;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.os.SystemClock;
import android.view.Surface;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Trip video with sound. The camera draws into the hardware H.264 encoder's surface;
 * microphone PCM (already captured for the voice) goes through an AAC encoder; one
 * muxer writes both into MP4. Timestamps of both are on the boot clock, so they line
 * up. Written in recording/, moved to the trips folder when finished.
 */
final class TripRecorder {
    static final int WIDTH = 1280, HEIGHT = 720, FPS = 30, BITRATE = 4_000_000;
    private static final int RATE = 16000, AUDIO_BITRATE = 64_000;
    private final MediaCodec video, audio;
    private final Surface surface;
    private final MediaMuxer muxer;
    private final AudioIO microphone;
    private final File partial, finished;
    private final Thread videoDrain, audioPump;
    private volatile boolean stopping;
    private int videoTrack = -1, audioTrack = -1;
    private boolean muxing;
    private long frames;
    private final Object muxLock = new Object();

    TripRecorder(File folder, int width, int height, AudioIO microphone) throws IOException {
        this.microphone = microphone;
        File recording = new File(folder, "recording");
        if (!recording.isDirectory() && !recording.mkdirs()) throw new IOException("trip folder");
        String name = "trip-" + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date()) + ".mp4";
        partial = new File(recording, name);
        finished = new File(folder, name);

        MediaFormat vf = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height);
        vf.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        vf.setInteger(MediaFormat.KEY_BIT_RATE, BITRATE);
        vf.setInteger(MediaFormat.KEY_FRAME_RATE, FPS);
        vf.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
        video = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
        video.configure(vf, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        surface = video.createInputSurface();

        MediaFormat af = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, RATE, 1);
        af.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC);
        af.setInteger(MediaFormat.KEY_BIT_RATE, AUDIO_BITRATE);
        af.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16384);
        audio = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC);
        audio.configure(af, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);

        muxer = new MediaMuxer(partial.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        // The phone stands upright; the front sensor is mounted at 270 degrees.
        muxer.setOrientationHint(270);
        video.start();
        audio.start();
        videoDrain = new Thread(() -> drain(video, true), "robotbrain-trip-video");
        audioPump = new Thread(this::pumpAudio, "robotbrain-trip-audio");
        videoDrain.setDaemon(true);
        audioPump.setDaemon(true);
        videoDrain.start();
        audioPump.start();
    }

    Surface surface() { return surface; }

    /** Feed microphone PCM into AAC with boot-clock timestamps; drain AAC into the muxer. */
    private void pumpAudio() {
        long cursor = microphone.cursor();
        boolean ended = false;
        try {
            while (true) {
                if (!stopping) {
                    AudioIO.PcmChunk chunk = microphone.readSince(cursor, 4096);
                    if (chunk.dropped) cursor = chunk.endCursor;
                    if (chunk.pcm.length > 0) {
                        cursor = chunk.endCursor;
                        long firstUs = chunk.frameElapsedMs * 1000L - chunk.pcm.length * 1_000_000L / (RATE * 2);
                        int offset = 0;
                        while (offset < chunk.pcm.length) {
                            int index = audio.dequeueInputBuffer(10_000);
                            if (index < 0) break;
                            ByteBuffer input = audio.getInputBuffer(index);
                            int n = Math.min(input.remaining(), chunk.pcm.length - offset);
                            n -= n & 1;
                            input.put(chunk.pcm, offset, n);
                            long pts = firstUs + offset * 1_000_000L / (RATE * 2);
                            audio.queueInputBuffer(index, 0, n, pts, 0);
                            offset += n;
                        }
                    }
                } else if (!ended) {
                    int index = audio.dequeueInputBuffer(10_000);
                    if (index >= 0) {
                        audio.queueInputBuffer(index, 0, 0, SystemClock.elapsedRealtimeNanos() / 1000,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        ended = true;
                    }
                }
                if (drainOnce(audio, false)) break;
                if (!stopping) SystemClock.sleep(10);
            }
        } catch (RuntimeException ignored) {
        }
    }

    private void drain(MediaCodec codec, boolean isVideo) {
        try {
            while (!drainOnce(codec, isVideo)) { /* until end of stream */ }
        } catch (RuntimeException ignored) {
        }
    }

    /** One output step; returns true at end of stream. */
    private boolean drainOnce(MediaCodec codec, boolean isVideo) {
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        int index = codec.dequeueOutputBuffer(info, isVideo ? 100_000 : 0);
        if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
            synchronized (muxLock) {
                if (isVideo) videoTrack = muxer.addTrack(codec.getOutputFormat());
                else audioTrack = muxer.addTrack(codec.getOutputFormat());
                if (videoTrack >= 0 && audioTrack >= 0 && !muxing) {
                    muxer.start();
                    muxing = true;
                }
            }
            return false;
        }
        if (index < 0) return false;
        ByteBuffer data = codec.getOutputBuffer(index);
        boolean config = (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0;
        synchronized (muxLock) {
            if (data != null && muxing && !config && info.size > 0) {
                data.position(info.offset);
                data.limit(info.offset + info.size);
                muxer.writeSampleData(isVideo ? videoTrack : audioTrack, data, info);
                if (isVideo) frames++;
            }
        }
        codec.releaseOutputBuffer(index, false);
        return (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
    }

    /** Finish the file; returns the finished path or null when nothing was recorded. */
    File stop() {
        stopping = true;
        try { video.signalEndOfInputStream(); } catch (RuntimeException ignored) { }
        try { videoDrain.join(3000); audioPump.join(3000); }
        catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        for (MediaCodec codec : new MediaCodec[]{video, audio}) {
            try { codec.stop(); } catch (RuntimeException ignored) { }
            codec.release();
        }
        surface.release();
        boolean ok;
        synchronized (muxLock) {
            ok = muxing && frames > 0;
            try { if (muxing) muxer.stop(); } catch (RuntimeException ignored) { ok = false; }
            muxer.release();
        }
        if (!ok) {
            //noinspection ResultOfMethodCallIgnored
            partial.delete();
            return null;
        }
        return partial.renameTo(finished) ? finished : null;
    }
}
