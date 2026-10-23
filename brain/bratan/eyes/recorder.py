"""Trip videos: while the body is moving, the frames the eyes already take go into MJPEG segments.

No extra camera work: only a JPEG encode (~3 ms) of frames that are processed anyway.
Segments of 2 minutes in /opt/robotbrain/bratan/videos; the oldest are deleted beyond
VIDEO_CAP_BYTES. The PC pulls finished segments (tools/pull_videos.py) and converts them.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

VIDEO_DIR = Path(os.environ.get('BRATAN_VIDEOS', '/opt/robotbrain/bratan/videos'))
SEGMENT_S = 120.
VIDEO_CAP_BYTES = 1_500_000_000
IDLE_CLOSE_S = 3.


class Recorder:
    def __init__(self, cv2):
        self.cv2 = cv2
        self.writer = None
        self.path = None
        self.opened = 0.
        self.last_frame = 0.
        self.fps = 8.

    def frame(self, image, moving, fps):
        now = time.time()
        if not moving:
            if self.writer is not None and now - self.last_frame > IDLE_CLOSE_S:
                self.close()
            return
        if self.writer is not None and now - self.opened > SEGMENT_S:
            self.close()
        if self.writer is None:
            self.open(image, fps)
        stamp = time.strftime('%H:%M:%S', time.localtime(now))
        self.cv2.putText(image, stamp, (8, 20), self.cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                         self.cv2.LINE_AA)
        self.writer.write(image)
        self.last_frame = now

    def open(self, image, fps):
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self.prune()
        h, w = image.shape[:2]
        self.fps = fps
        (VIDEO_DIR / 'recording').mkdir(exist_ok=True)
        self.path = VIDEO_DIR / 'recording' / time.strftime('trip-%Y%m%d-%H%M%S.avi')
        self.writer = self.cv2.VideoWriter(str(self.path), self.cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError('video_writer_not_opened')
        self.opened = time.time()

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        if self.path and self.path.exists():
            self.path.rename(VIDEO_DIR / self.path.name)          # finished: ready to pull
        self.path = None

    def prune(self):
        files = sorted(VIDEO_DIR.glob('trip-*.avi'), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > VIDEO_CAP_BYTES:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink()
