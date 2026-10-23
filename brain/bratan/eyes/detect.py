"""YOLO object detection, face identity and a short-term IoU tracker."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
RU = {'person': 'человек', 'dog': 'собака', 'cat': 'кошка', 'chair': 'стул', 'couch': 'диван', 'bed': 'кровать',
      'dining table': 'стол', 'tv': 'телевизор', 'laptop': 'ноутбук', 'potted plant': 'растение',
      'refrigerator': 'холодильник', 'book': 'книга', 'bottle': 'бутылка', 'cup': 'чашка', 'backpack': 'рюкзак',
      'suitcase': 'чемодан', 'teddy bear': 'игрушка', 'sports ball': 'мяч', 'clock': 'часы', 'vase': 'ваза',
      'handbag': 'сумка', 'cell phone': 'телефон', 'keyboard': 'клавиатура', 'mouse': 'мышка', 'bowl': 'миска',
      'oven': 'духовка', 'microwave': 'микроволновка', 'sink': 'раковина', 'toilet': 'унитаз', 'umbrella': 'зонт',
      'remote': 'пульт', 'cabinet': 'шкаф', 'shelf': 'полка', 'wardrobe': 'шкаф для одежды', 'drawer': 'комод', 'table': 'стол', 'lamp': 'лампа', 'plant': 'растение', 'box': 'коробка', 'basket': 'корзина', 'pillow': 'подушка', 'blanket': 'плед', 'mirror': 'зеркало', 'picture': 'картина', 'desk': 'письменный стол', 'curtain': 'штора', 'rug': 'ковёр', 'shoe': 'обувь', 'toy': 'игрушка', 'bag': 'сумка', 'scissors': 'ножницы', 'bicycle': 'велосипед', 'bench': 'скамейка', 'tie': 'галстук',
      # names the old robot's open-vocabulary detector used (08.10: «speaker», «photo» in the mind's own words)
      'speaker': 'колонка', 'photo': 'фотография', 'television': 'телевизор', 'towel': 'полотенце',
      'armchair': 'кресло', 'parcel': 'посылка', 'package': 'посылка', 'cardboard box': 'коробка',
      'paper': 'бумаги', 'doorway': 'дверной проём', 'door': 'дверь', 'window': 'окно', 'notebook': 'блокнот',
      'magazine': 'журнал', 'newspaper': 'газета', 'document': 'документ', 'card': 'открытка',
      'picture frame': 'рамка с фото', 'shopping bag': 'пакет', 'plastic bag': 'пакет', 'wallet': 'кошелёк',
      'keys': 'ключи', 'glasses': 'очки', 'slipper': 'тапок', 'clothing': 'одежда', 'sock': 'носок',
      'ball': 'мяч', 'remote control': 'пульт', 'smartphone': 'телефон', 'tablet': 'планшет',
      'computer mouse': 'мышка', 'charger': 'зарядка', 'charging cable': 'провод зарядки',
      'power bank': 'пауэрбанк', 'camera': 'фотоаппарат', 'mug': 'кружка', 'glass': 'стакан', 'plate': 'тарелка',
      'fork': 'вилка', 'spoon': 'ложка', 'knife': 'нож', 'food': 'еда', 'fruit': 'фрукты', 'apple': 'яблоко',
      'banana': 'банан', 'bread': 'хлеб', 'pet bowl': 'миска Таффы', 'trash can': 'мусорное ведро',
      'flower pot': 'цветочный горшок', 'robot vacuum': 'робот-пылесос', 'scooter': 'самокат',
      'door handle': 'дверная ручка', 'stairs': 'лестница', 'charging dock': 'зарядная база'}
THRESHOLD = {'person': .45, 'dog': .40, 'cat': .40}
DEFAULT_THRESHOLD = .45


def session(cv2_unused, path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


class Detector:
    def __init__(self, cv2, model: Path, threads=3, names=None):
        self.cv2 = cv2
        self.session = session(cv2, model, threads)
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[-1] if isinstance(self.input.shape[-1], int) else 640
        self.names = names or dict(enumerate(COCO))

    def __call__(self, image):
        h, w = image.shape[:2]
        gain = min(self.size / w, self.size / h)
        nw, nh = round(w * gain), round(h * gain)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = self.cv2.resize(image, (nw, nh), interpolation=self.cv2.INTER_LINEAR)
        tensor = (canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)) * (1 / 255.)
        rows = self.session.run(None, {self.input.name: tensor})[0][0]
        out = []
        for row in rows:
            score, cls = float(row[4]), int(round(float(row[5])))
            label = self.names.get(cls)
            if label is None or score < THRESHOLD.get(label, DEFAULT_THRESHOLD):
                continue
            box = [(float(row[0]) - dx) / gain, (float(row[1]) - dy) / gain,
                   (float(row[2]) - dx) / gain, (float(row[3]) - dy) / gain]
            box = [min(max(0., box[0]), w), min(max(0., box[1]), h), min(max(0., box[2]), w), min(max(0., box[3]), h)]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            out.append({'label': label, 'score': round(score, 3), 'box': [round(v, 1) for v in box]})
        return _nms(out)


# The camera on the bow looks up 21°: the top quarter of the upright 480x640 frame is ceiling. The 320 px detector
# sees the floor and what stands on it larger without it (10.10 evening, 2846 trip frames made like the eyes':
# people in 60 frames instead of 36, Taffa in 88 instead of 81 — the same one inference).
CROP_TOP = .25


def detect_lower(detector, image, top=CROP_TOP):
    """Detections on the frame without its top `top`, in full-frame coordinates. A person cut by the crop goes on
    above it: the box reaches the top of the frame, where the face is (faces are matched to person boxes)."""
    cut = int(image.shape[0] * top)
    if cut <= 0:
        return detector(image)
    items = detector(image[cut:])
    for item in items:
        x1, y1, x2, y2 = item['box']
        if item['label'] == 'person' and y1 <= 2:
            y1 = -cut
        item['box'] = [x1, round(y1 + cut, 1), x2, round(y2 + cut, 1)]
    return items


def merge_pet_labels(items, threshold=.5):
    """No cat at home: a «cat» is our dog seen badly (the 320 px detector called Taffa a cat in 15 of 40 frames,
    09.10) — so the mind and the body did not see him. Relabelled; dropped where a dog box already covers it."""
    dogs = [i for i in items if i['label'] == 'dog']
    out = []
    for item in items:
        if item['label'] == 'cat':
            if any(_iou(item['box'], d['box']) > threshold for d in dogs):
                continue
            item = {**item, 'label': 'dog', 'seen_as': 'cat'}
        out.append(item)
    return out


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def _nms(items, threshold=.6):
    kept = []
    for item in sorted(items, key=lambda i: -i['score']):
        if all(item['label'] != k['label'] or _iou(item['box'], k['box']) < threshold for k in kept):
            kept.append(item)
    return kept


LEARN_MIN = .30          # an unrecognised angle this close to the track's own person may be learned
LEARN_MARGIN = .05      # ... and this much closer to them than to anyone else
LEARN_EVERY_S = 30.     # one learned sample per person this often
LEARNED_KEPT = 20       # learned samples kept per person (the enrolled ones are never pushed out by them)


class Faces:
    """YuNet detection + SFace embeddings matched against faces.json."""

    def __init__(self, cv2, models: Path, db: Path):
        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(models / 'identity/face_detection_yunet_2023mar.onnx'),
                                                  '', (320, 320), .85)
        self.recognizer = cv2.FaceRecognizerSF.create(str(models / 'identity/face_recognition_sface_2021dec.onnx'), '')
        self.db = db
        self.mtime = None
        self.people: dict[str, np.ndarray] = {}
        self.learned_at: dict[str, float] = {}

    def _load(self):
        try:
            mtime = self.db.stat().st_mtime
        except OSError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        data = json.loads(self.db.read_text(encoding='utf-8')).get('people', {})
        self.people = {}
        for name, person in data.items():
            samples = np.asarray(list(person.get('samples', [])) + list(person.get('learned', [])), dtype=np.float32)
            if samples.ndim == 2 and len(samples):
                self.people[name] = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-9)

    def __call__(self, image):
        self._load()
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        out = []
        for face in faces if faces is not None else []:
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).reshape(-1)
            feature = feature / (np.linalg.norm(feature) + 1e-9)
            scores = sorted(((float(np.max(s @ feature)), n) for n, s in self.people.items()), reverse=True)
            name = None
            if scores and scores[0][0] >= .40 and (len(scores) == 1 or scores[0][0] - scores[1][0] > .05):
                name = scores[0][1]
            x, y, fw, fh = (float(v) for v in face[:4])
            out.append({'box': [x, y, x + fw, y + fh], 'name': name,
                        'similarity': round(scores[0][0], 3) if scores else None, 'embedding': feature,
                        'best': scores[0][1] if scores else None,
                        'margin': round(scores[0][0] - scores[1][0], 3) if len(scores) > 1 else 1.})
        return out

    def may_learn(self, name, face, now):
        """An unrecognised face on a track that a sure face named `name`: worth learning if it is still closest
        to `name`, clearly, and not too far off."""
        return (face.get('name') is None and face.get('best') == name and (face.get('similarity') or 0) >= LEARN_MIN
                and (face.get('margin') or 0) >= LEARN_MARGIN and now - self.learned_at.get(name, -1e9) >= LEARN_EVERY_S)

    def learn(self, name, embedding, now):
        """Keep one more look of a known person (a new angle, other light) next to the enrolled ones."""
        if not self.db.exists():
            return False
        data = json.loads(self.db.read_text(encoding='utf-8'))
        person = data.get('people', {}).get(name)
        if person is None:
            return False
        self.learned_at[name] = now
        person['learned'] = (list(person.get('learned', [])) + [list(map(float, embedding))])[-LEARNED_KEPT:]
        person['learned_at'] = time.time()
        temporary = self.db.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        temporary.replace(self.db)
        self.mtime = None
        return True

    def enroll(self, name, embeddings):
        data = json.loads(self.db.read_text(encoding='utf-8')) if self.db.exists() else {'schema': 1, 'people': {}}
        person = data['people'].setdefault(name, {'schema': 1, 'name': name, 'enrolled_at': time.time(), 'samples': []})
        person['samples'] = (person['samples'] + [list(map(float, e)) for e in embeddings])[-12:]
        temporary = self.db.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        temporary.replace(self.db)
        self.mtime = None


class Tracker:
    """IoU association across frames; names stick to a track once recognised."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, items, now):
        self.tracks = {k: t for k, t in self.tracks.items() if now - t['seen'] < 2.}
        used = set()
        for item in items:
            best, best_iou = None, .3
            for key, track in self.tracks.items():
                if key in used or track['label'] != item['label']:
                    continue
                overlap = _iou(track['box'], item['box'])
                if overlap > best_iou:
                    best, best_iou = key, overlap
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {'label': item['label'], 'first': now, 'frames': 0, 'name': None}
            used.add(best)
            track = self.tracks[best]
            track.update(box=item['box'], seen=now)
            track['frames'] += 1
            if item.get('name'):
                track['name'] = item['name']
            elif track['name']:
                # An unrecognised angle of the face on a named track is the same person (11.10: «человек», then
                # «Сергей», then «человек» again as he turned his head).
                item['name'] = track['name']
                item['name_kept'] = True
            item['track'] = best
            item['frames'] = track['frames']
        return items
