"""Offline training helpers only; extracted from the original training snapshot."""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

def read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    offset = 0

    def token() -> bytes:
        nonlocal offset
        while offset < len(raw):
            if raw[offset:offset + 1] == b"#":
                offset = raw.find(b"\n", offset) + 1
            elif raw[offset:offset + 1].isspace():
                offset += 1
            else:
                break
        end = offset
        while end < len(raw) and not raw[end:end + 1].isspace():
            end += 1
        value = raw[offset:end]
        offset = end
        return value

    if token() != b"P5":
        raise ValueError("unsupported_pgm")
    width, height, maximum = int(token()), int(token()), int(token())
    if maximum != 255 or not 1 <= width * height <= 20_000_000:
        raise ValueError("invalid_pgm")
    while offset < len(raw) and raw[offset:offset + 1].isspace():
        offset += 1
    pixels = np.frombuffer(raw[offset:], dtype=np.uint8)
    if pixels.size != width * height:
        raise ValueError("pgm_size_mismatch")
    return pixels.reshape((height, width))

def read_map_yaml(path: Path) -> tuple[Path, float, tuple[float, float, float]]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    image = values["image"].strip("'\"")
    origin = tuple(float(part.strip()) for part in values["origin"].strip("[]").split(","))
    if len(origin) != 3 or abs(origin[2]) > 1e-9:
        raise ValueError("rotated_map_origin_unsupported")
    return (path.parent / image).resolve(), float(values["resolution"]), origin

class DistanceMap:

    def __init__(self, yaml_path: Path):
        (image, self.resolution, self.origin) = read_map_yaml(yaml_path)
        pixels = read_pgm(image)
        occupied = pixels <= 65
        if occupied.sum() < 20:
            raise ValueError('map_has_too_few_obstacles')
        self.field = distance_transform_edt(~occupied).astype(np.float32) * self.resolution
        (self.height, self.width) = pixels.shape
        self.free = pixels >= 250

    def distances(self, pose: np.ndarray, points: np.ndarray) -> np.ndarray:
        (c, s) = (math.cos(float(pose[2])), math.sin(float(pose[2])))
        world_x = float(pose[0]) + c * points[:, 0] - s * points[:, 1]
        world_y = float(pose[1]) + s * points[:, 0] + c * points[:, 1]
        columns = (world_x - self.origin[0]) / self.resolution
        rows = self.height - 1.0 - (world_y - self.origin[1]) / self.resolution
        return map_coordinates(self.field, np.vstack((rows, columns)), order=1, mode='constant', cval=0.5, prefilter=False)
