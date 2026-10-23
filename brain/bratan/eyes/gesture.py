"""Body language: a wave, hands up (YOLO26n-pose keypoints), and a smile or a sad face (FER+).

Loona and EMO answer a wave; Pepper and Moxie mirror a smile. Both run only when someone is in view, in a
worker so the camera loop is never held up: pose ~4 times a second, the face expression on the faces the
recogniser already found.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

# COCO keypoints
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10
KP_MIN = .4
WAVE_WINDOW_S = 2.5
WAVE_TURNS = 2          # the hand changed direction at least this often, above the shoulder
GESTURE_GAP_S = 6.
FER_LABELS = ('neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt')


class Pose:
    def __init__(self, model_path, threads=2):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[-1] if isinstance(self.input.shape[-1], int) else 320

    def __call__(self, cv2, image, threshold=.45):
        """[{'box', 'score', 'kp': (17, 3) array in image pixels}] for the people found."""
        h, w = image.shape[:2]
        gain = min(self.size / w, self.size / h)
        nw, nh = round(w * gain), round(h * gain)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.
        rows = self.session.run(None, {self.input.name: tensor})[0][0]
        people = []
        for row in rows:
            if float(row[4]) < threshold:
                continue
            kp = np.asarray(row[6:6 + 51], dtype=float).reshape(17, 3)
            kp[:, 0] = (kp[:, 0] - dx) / gain
            kp[:, 1] = (kp[:, 1] - dy) / gain
            box = [(float(row[0]) - dx) / gain, (float(row[1]) - dy) / gain,
                   (float(row[2]) - dx) / gain, (float(row[3]) - dy) / gain]
            people.append({'box': box, 'score': float(row[4]), 'kp': kp})
        return people


class Gestures:
    """A wave (a hand above the shoulder going side to side) and both hands up, per tracked person."""

    def __init__(self):
        self.hands: dict[int, deque] = {}       # track -> (time, wrist x relative, above shoulder) per hand
        self.said: dict[tuple, float] = {}

    def update(self, people, now=None):
        """`people`: [{'track', 'kp', 'box', ...}]. Returns [(kind, person)] newly seen."""
        now = time.monotonic() if now is None else now
        found = []
        for person in people:
            kp, track = person['kp'], person.get('track', 0)
            ls, rs = kp[L_SHOULDER], kp[R_SHOULDER]
            if ls[2] < KP_MIN or rs[2] < KP_MIN:
                continue
            width = max(20., abs(ls[0] - rs[0]))
            shoulder_y = (ls[1] + rs[1]) / 2
            up = [kp[w][2] >= KP_MIN and kp[w][1] < shoulder_y - .1 * width for w in (L_WRIST, R_WRIST)]
            nose = kp[NOSE]
            if all(up) and nose[2] >= KP_MIN and min(kp[L_WRIST][1], kp[R_WRIST][1]) < nose[1]:
                if self._fresh(('hands_up', track), now):
                    found.append(('hands_up', person))
                continue
            history = self.hands.setdefault(track, deque(maxlen=40))
            for side, wrist in ((0, L_WRIST), (1, R_WRIST)):
                if up[side]:
                    history.append((now, side, (kp[wrist][0] - (ls[0] + rs[0]) / 2) / width))
            recent = [h for h in history if now - h[0] < WAVE_WINDOW_S]
            for side in (0, 1):
                xs = [x for t, s, x in recent if s == side]
                if len(xs) < 5:
                    continue
                turns, direction = 0, 0
                for a, b in zip(xs, xs[1:]):
                    step = b - a
                    if abs(step) < .08:
                        continue
                    sign = 1 if step > 0 else -1
                    if direction and sign != direction:
                        turns += 1
                    direction = sign
                if turns >= WAVE_TURNS and max(xs) - min(xs) > .35 and self._fresh(('wave', track), now):
                    found.append(('wave', person))
                    history.clear()
                    break
        for track in list(self.hands):
            if not self.hands[track] or now - self.hands[track][-1][0] > 10:
                self.hands.pop(track, None)
        return found

    def _fresh(self, key, now):
        if now - self.said.get(key, -1e9) < GESTURE_GAP_S:
            return False
        self.said[key] = now
        return True


class Expressions:
    """FER+ (ONNX model zoo): the face's expression; a smile or sadness held for a few frames is news."""

    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.recent: dict[str, deque] = {}
        self.said: dict[tuple, float] = {}

    def __call__(self, cv2, image, box):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        pad = int(.1 * (x2 - x1))
        h, w = image.shape[:2]
        crop = image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return None
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        scores = self.session.run(None, {self.input: gray[None, None]})[0].reshape(-1)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        return dict(zip(FER_LABELS, probs.round(3).tolist()))

    def update(self, who, probs, now=None):
        """Returns 'smile' / 'sad' / 'surprise' when held over a few frames (each once in a while)."""
        now = time.monotonic() if now is None else now
        history = self.recent.setdefault(who or '?', deque(maxlen=6))
        history.append(probs)
        if len(history) < 3:
            return None
        mean = {k: float(np.mean([p[k] for p in list(history)[-3:]])) for k in FER_LABELS}
        # A face seen from the floor reads sadder than it is: sadness must be clear before it is news.
        for kind, label, level, gap in (('smile', 'happiness', .6, 60.), ('sad', 'sadness', .6, 600.),
                                        ('surprise', 'surprise', .7, 120.)):
            if mean[label] >= level and now - self.said.get((who, kind), -1e9) > gap:
                self.said[(who, kind)] = now
                return kind
        return None
