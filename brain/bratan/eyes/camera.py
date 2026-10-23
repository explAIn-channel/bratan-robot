"""Front camera frames from the face app (raw NV21: no JPEG encode/decode) and geometry."""
from __future__ import annotations

import math
import time

import numpy as np

from ..common.config import ASSETS
from ..face import client as face


def boottime():
    return time.clock_gettime(getattr(time, 'CLOCK_BOOTTIME', 7))


def fetch(cv2):
    """Latest frame as an upright BGR image (480x640) with metadata, or None."""
    before, before_boot = time.monotonic(), boottime()
    with face.request('/camera/frame.nv21', 'camera.read', timeout=1.5) as response:
        raw = response.read()
        headers = response.headers
    after, after_boot = time.monotonic(), boottime()
    width = int(headers['X-RobotBrain-Frame-Width'])
    height = int(headers['X-RobotBrain-Frame-Height'])
    if len(raw) != width * height * 3 // 2:
        raise ValueError('nv21_size')
    yuv = np.frombuffer(raw, np.uint8).reshape(height * 3 // 2, width)
    image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    sensor_s = int(headers['X-RobotBrain-Sensor-Timestamp-Ns']) / 1e9
    # Sensor time is CLOCK_BOOTTIME; move it into CLOCK_MONOTONIC.
    mono = sensor_s + (before + after - before_boot - after_boot) / 2
    return image, {'mono': mono, 'seq': int(headers['X-RobotBrain-Frame-Sequence']),
                   'age': after - mono}


class Geometry:
    """Pinhole + distortion for the upright 480x640 image; camera mounted on the bow."""

    def __init__(self, cv2):
        import yaml
        cfg = yaml.safe_load((ASSETS / 'camera/front_camera_info.yaml').read_text())
        self.cv2 = cv2
        self.K = np.array(cfg['camera_matrix']['data'], dtype=float).reshape(3, 3)
        self.D = np.array(cfg['distortion_coefficients']['data'], dtype=float)
        self.width, self.height = cfg['image_width'], cfg['image_height']
        self.position = np.array([.172872511941, -.0178, .2])
        roll, pitch = math.radians(1.94467368), math.radians(-20.96501325)
        self.rotation = _rpy(roll, pitch, 0.) @ _rpy(-math.pi / 2, 0., -math.pi / 2)

    def ray(self, u, v):
        """Unit ray in the base frame through pixel (u, v)."""
        p = self.cv2.undistortPoints(np.array([[[u, v]]], dtype=float), self.K, self.D).reshape(2)
        r = self.rotation @ np.array([p[0], p[1], 1.])
        return r / np.linalg.norm(r)

    def bearing(self, u, v=None):
        r = self.ray(u, self.height / 2 if v is None else v)
        return math.atan2(r[1], r[0])

    def floor(self, u, v):
        """Base-frame floor point seen at pixel (u, v), or None above the horizon."""
        r = self.ray(u, v)
        if r[2] >= -1e-3:
            return None
        hit = self.position - r * (self.position[2] / r[2])
        return (float(hit[0]), float(hit[1])) if .15 < hit[0] < 6. else None


def _rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])
