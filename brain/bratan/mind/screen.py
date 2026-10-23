"""Cards on the face screen: countdown digits, a word, a photo (the face app's /photo overlay).

The face shows an overlay JPEG for up to 30 s, then its eyes again. Text is drawn with FreeType (DejaVu has
Cyrillic; OpenCV's own fonts do not), big and centred, light on a dark background — readable from across
the room.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
SIZE = (960, 540)
_ft = None


def _freetype(cv2):
    global _ft
    if _ft is None:
        _ft = cv2.freetype.createFreeType2()
        _ft.loadFontData(str(FONT), 0)
    return _ft


def card(text, sub='', *, fg=(255, 255, 255), bg=(28, 22, 18), accent=(0, 200, 255)):
    """A JPEG with `text` as big as fits, `sub` smaller under it."""
    import cv2
    w, h = SIZE
    image = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (w - 13, h - 13), accent, 6)
    text = str(text)
    try:
        ft = _freetype(cv2)

        def fit(line, start, limit):
            size = start
            while size > 24:
                (tw, th), _ = ft.getTextSize(line, size, -1)
                if tw <= limit:
                    return size, tw, th
                size -= 6
            (tw, th), _ = ft.getTextSize(line, size, -1)
            return size, tw, th
        start = 300 if len(text) <= 3 else 150 if len(text) <= 10 else 100
        size, tw, th = fit(text, start, w - 80)
        base = int(h * (.62 if not sub else .52)) if len(text) > 3 else int(h * (.78 if not sub else .66))
        ft.putText(image, text, ((w - tw) // 2, base - th // 2 + th // 2), size, fg, -1, cv2.LINE_AA, True)
        if sub:
            sub_size, sw, sh = fit(sub, 56, w - 80)
            ft.putText(image, sub, ((w - sw) // 2, min(h - 40, base + 50 + sh)), sub_size, accent, -1,
                       cv2.LINE_AA, True)
    except Exception:                         # no FreeType: digits and Latin still work with OpenCV's font
        cv2.putText(image, text, (60, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 6, fg, 12, cv2.LINE_AA)
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return jpeg.tobytes() if ok else b''


def weather_card(current):
    """Vector-style weather on the face: a drawn sky (sun, cloud, rain, snow) and the temperature."""
    import cv2
    w, h = SIZE
    code = current.get('code') or 0
    sunny, rainy, snowy = code in (0, 1), code in (51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99), code in (71, 73, 75,
                                                                                                         77, 85, 86)
    image = np.full((h, w, 3), (60, 35, 20) if not sunny else (90, 60, 25), dtype=np.uint8)
    cx, cy = 230, 250
    if sunny or code == 2:
        cv2.circle(image, (cx - 30, cy - 30), 90, (0, 200, 255), -1, cv2.LINE_AA)
        for k in range(12):
            a = k * np.pi / 6
            cv2.line(image, (int(cx - 30 + 115 * np.cos(a)), int(cy - 30 + 115 * np.sin(a))),
                     (int(cx - 30 + 150 * np.cos(a)), int(cy - 30 + 150 * np.sin(a))), (0, 200, 255), 10, cv2.LINE_AA)
    if not sunny:
        for dx, dy, r in ((-60, 20, 70), (20, -10, 90), (100, 25, 65), (20, 45, 75)):
            cv2.circle(image, (cx + dx, cy + dy), r, (235, 235, 235), -1, cv2.LINE_AA)
    if rainy:
        for k in range(7):
            x = cx - 120 + k * 40
            cv2.line(image, (x, cy + 120), (x - 20, cy + 190), (255, 160, 60), 8, cv2.LINE_AA)
    if snowy:
        for k in range(7):
            cv2.circle(image, (cx - 120 + k * 40, cy + 150 + (k % 2) * 30), 10, (255, 255, 255), -1, cv2.LINE_AA)
    temp = current.get('temp')
    big = f'{temp:+.0f}°' if isinstance(temp, (int, float)) else '?'
    small = current.get('words') or ''
    if isinstance(current.get('low'), (int, float)) and isinstance(current.get('high'), (int, float)):
        small = (small + f', днём {current["low"]:+.0f}…{current["high"]:+.0f}°').strip(', ')
    try:
        ft = _freetype(cv2)
        ft.putText(image, big, (470, 300), 190, (255, 255, 255), -1, cv2.LINE_AA, True)
        (sw, sh), _ = ft.getTextSize(small, 44, -1)
        size = 44 if sw < w - 60 else 34
        ft.putText(image, small, (40, h - 40), size, (0, 230, 255), -1, cv2.LINE_AA, True)
    except Exception:
        cv2.putText(image, big.replace('°', ''), (470, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return jpeg.tobytes() if ok else b''


def show(jpeg, seconds=3):
    from ..face import client as face
    if jpeg:
        face.post(f'/photo?seconds={max(1, min(30, int(round(seconds))))}', 'face.write', jpeg, 3., 'image/jpeg')


def clear():
    from ..face import client as face
    face.post('/photo/clear', 'face.write')
