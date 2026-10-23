"""Eyes process: camera -> detector -> faces -> tracker -> bus, as fast as the mode allows.

Bus in:  body.state            pose history (map coordinates of what is seen)
         mind.eyes {mode}      'focus' (conversation/tracking), 'normal', 'rest' (night)
         eyes.look {id}        save the current frame + a full-resolution object list
         eyes.enroll {name}    store the recently seen unknown face under a name
Bus out: eyes.seen {mono, items, ...} every processed frame
         eyes.looked {id, path, items}, eyes.enrolled {name, samples}
         eyes.things {items}  small things and where they lie (now and then)
         eyes.gesture {kind: wave|hands_up|smile|sad|surprise, who, ...}  body language of whoever is in view
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

from ..common import bus as busmod
from ..common.config import MODELS, STATE, load, pin
from ..common.geometry import compose, interpolate

RATES = {'focus': 12., 'active': 8., 'attentive': 6., 'idle': 2., 'rest': .2}
ATTENTIVE_S = 5.           # someone in view this recently: a quicker look (a wave needs ~5 frames a second)
FACE_PERIOD_S = .25
RUN_DIR = Path(os.environ.get('BRATAN_RUN', '/run/bratan'))


def import_cv2():
    faces_site = Path('/opt/robotbrain/venvs/faces/lib/python3.12/site-packages')
    if faces_site.is_dir() and str(faces_site) not in sys.path:
        sys.path.insert(0, str(faces_site))
    import cv2
    if str(faces_site) in sys.path:
        sys.path.remove(str(faces_site))
    cv2.setNumThreads(2)
    return cv2


RECORD_TAIL_S = 10.
# Small household things (YOLOE household, ~100 classes; 0.4 s a frame on two big cores, so only now and then):
# «где мои ключи?» (owner's home, 08.10). Furniture is what they lie on.
SURVEY_S = {'focus': 6., 'active': 4., 'attentive': 6., 'idle': 10.}
SURVEY_MIN_LIGHT = 35
SMALL_THINGS = {'keys', 'glasses', 'wallet', 'remote control', 'smartphone', 'tablet', 'laptop', 'book', 'notebook',
                'cup', 'mug', 'bottle', 'glass', 'backpack', 'handbag', 'shopping bag', 'toy', 'ball', 'slipper',
                'shoe', 'pet bowl', 'charger', 'power bank', 'camera', 'scissors', 'umbrella', 'plate', 'bowl',
                'computer mouse', 'keyboard', 'clock', 'magazine', 'newspaper', 'document', 'box', 'cardboard box',
                'package', 'parcel', 'towel', 'clothing', 'sock', 'flower pot', 'plant', 'fruit', 'apple', 'banana'}
SUPPORTS = {'table': 'на столе', 'desk': 'на письменном столе', 'bed': 'на кровати', 'couch': 'на диване',
            'armchair': 'на кресле', 'chair': 'на стуле', 'shelf': 'на полке', 'cabinet': 'на шкафу',
            'drawer': 'на комоде', 'sink': 'у раковины'}

# Body language (gesture.py): pose (93 ms on the phone) up to 5 times a second while someone is in view,
# the face's expression (FER+, 105 ms a face) once a second, only for a face big enough to read.
GESTURE_PERIOD_S = .2
GESTURE_MIN_PERSON_PX = 120         # a far-off figure: too small for hands to be read
EXPRESSION_PERIOD_S = 1.
EXPRESSION_MIN_FACE_PX = 56
GESTURE_GIVE_UP = 5                 # errors in a row: the models are missing or broken, stop trying


class Eyes:
    def __init__(self, config, bus_address=busmod.DEFAULT_ADDRESS):
        self.config = config
        self.cv2 = cv2 = import_cv2()
        from .camera import Geometry
        from .detect import Detector, Faces, Tracker
        self.geometry = Geometry(cv2)
        self.detector = Detector(cv2, MODELS / 'yolo26n-track320.onnx', threads=3)
        self.wide = None                    # 640 model, loaded on first `look`
        self.survey = None                  # YOLOE household, loaded on first survey
        self.survey_at = 0.
        self.survey_busy = False
        from concurrent.futures import ThreadPoolExecutor
        self.survey_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='survey')
        self.pose = self.gestures = self.expressions = None     # loaded when someone is first in view
        self.gesture_at = self.expression_at = 0.
        self.gesture_busy = False
        self.gesture_errors = 0
        self.gesture_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gesture')
        self.last_people = []
        self.person_at = -1e9
        self.faces = Faces(cv2, MODELS, Path('/opt/robotbrain/state/faces.json'))
        self.tracker = Tracker()

        self.poses = deque(maxlen=200)      # (mono, map pose)
        self.body = {}
        self.mode_override = None
        self.override_until = 0.
        self.requests = deque(maxlen=8)
        self.unknown = deque(maxlen=12)     # (mono, embedding) of unrecognised faces
        self.last_face = 0.
        self.last_faces = []
        self.frame = None
        self.stats = {'frames': 0, 'detect_ms': 0., 'faces_ms': 0., 'latency_ms': 0.}
        self.storage = 'ok'
        self.bus = busmod.Client('eyes', ('body.state', 'mind.eyes', 'eyes.look', 'eyes.enroll', 'system.storage.state'),
                                 self.on_message, bus_address)
        self.stopping = threading.Event()

    def on_message(self, topic, data):
        if topic == 'system.storage.state':
            self.storage = data.get('level', 'ok')
            return
        if topic == 'body.state':
            self.body = data
            if data.get('localized') and data.get('mono') is not None:
                if not self.poses or data['mono'] > self.poses[-1][0]:
                    self.poses.append((data['mono'], tuple(data['pose'])))
        elif topic == 'mind.eyes':
            self.mode_override = data.get('mode')
            self.override_until = time.monotonic() + float(data.get('seconds', 600))
        elif topic in ('eyes.look', 'eyes.enroll'):
            self.requests.append((topic, data))

    def mode(self):
        if self.mode_override and time.monotonic() < self.override_until:
            return self.mode_override
        body = self.body
        if body.get('skill') or body.get('moving'):
            return 'focus' if (body.get('skill') or {}).get('kind') == 'track' else 'active'
        if time.monotonic() - self.person_at < ATTENTIVE_S:
            return 'attentive'
        return 'idle'

    def run(self):
        from .camera import fetch
        self.bus.start()
        last_seq = None
        camera_rate = None
        while not self.stopping.is_set():
            mode = self.mode()
            period = 1. / RATES.get(mode, 2.)
            if RATES.get(mode) != camera_rate:
                # The camera produces only what we use: fresh frames, no wasted ISP work.
                try:
                    from ..face import client as face
                    face.post(f'/camera/rate?fps={RATES.get(mode, 2.):.1f}', 'camera.read')
                    camera_rate = RATES.get(mode)
                except Exception:
                    pass
            started = time.monotonic()
            try:
                self._requests()
                frame = fetch(self.cv2)
                if frame[1]['seq'] != last_seq:
                    last_seq = frame[1]['seq']
                    self.frame = frame
                    self.process(*frame)
                    self.trip_video()
                    self.maybe_survey(frame)
                    self.maybe_gesture(frame)
            except Exception as error:
                if '503' in str(error) and (time.monotonic() - getattr(self, 'record_switched', -1e9) < 3.
                                            or self.mode() == 'rest'):
                    # Recorder switching, or the night rest rate (a frame per 5 s is older than the app's 2 s).
                    self.stopping.wait(.2)      # the camera is rebuilding its session for the recorder
                    continue
                self.bus.publish('eyes.error', {'error': f'{type(error).__name__}:{error}'[:160]})
                self.stopping.wait(1.)
            self.stopping.wait(max(0., period - (time.monotonic() - started)))

    def trip_video(self):
        """720p30 video with sound, recorded by the face app's hardware encoder only while driving."""
        if self.body.get('moving'):
            self.moved_at = time.monotonic()
        # A pause between moves (arrived, now looking around) keeps the same file:
        # every switch rebuilds the camera session and leaves the eyes without frames briefly.
        wanted = time.monotonic() - getattr(self, 'moved_at', -1e9) < RECORD_TAIL_S
        if self.storage == 'critical':
            wanted = False                       # the phone is almost full: no new videos
        if wanted != getattr(self, 'recording', False):
            try:
                from ..face import client as face
                face.post(f'/camera/record?on={int(wanted)}', 'camera.read')
                self.recording = wanted
                self.record_switched = time.monotonic()
                if not wanted:
                    prune_trips(TRIPS_CAP_LOW if self.storage != 'ok' else TRIPS_CAP)
            except Exception as error:
                self.bus.publish('eyes.error', {'error': f'record:{type(error).__name__}:{error}'[:160]})

    def process(self, image, meta):
        t0 = time.monotonic()
        from .detect import detect_lower
        items = detect_lower(self.detector, image)
        pets = self.config.get('pets', {})
        if pets.get('dog') and not pets.get('cat'):
            from .detect import merge_pet_labels
            items = merge_pet_labels(items)
        t1 = time.monotonic()
        people = [i for i in items if i['label'] == 'person']
        # On the dock nobody is recognised (owner 08.10 21:30: «на базе видит меня и пытается заговорить — не
        # нужно»): faces only off the dock, or when someone has started a talk (the eyes are then in focus).
        if people and not self.resting() and t1 - self.last_face >= FACE_PERIOD_S:
            self.last_face = t1
            self.last_faces = self.faces(image)
            self.stats['faces_ms'] = round((time.monotonic() - t1) * 1000, 1)
            for face in self.last_faces:
                if face['name'] is None:
                    self.unknown.append((t1, face['embedding']))
        self._assign_faces(people)
        self.tracker.update(items, meta['mono'])
        if len(people) == 1 and people[0].get('name_kept') and people[0].get('face_seen') is not None:
            face = people[0]['face_seen']
            if self.faces.may_learn(people[0]['name'], face, t1):
                try:
                    if self.faces.learn(people[0]['name'], face['embedding'], t1):
                        self.bus.publish('eyes.learned', {'name': people[0]['name'],
                                                          'similarity': face.get('similarity')})
                except Exception as error:
                    self.bus.publish('eyes.error', {'error': f'learn:{type(error).__name__}:{error}'[:160]})
        for person in people:
            person.pop('face_seen', None)
        pose = interpolate(self.poses, meta['mono'])
        for item in items:
            self._locate(item, pose, image.shape)
            item.pop('embedding', None)
        # Someone who stays a few frames (not a coat on a chair flickering into a «person»).
        self.last_people = [dict(p) for p in people if p.get('score', 0) >= .5 and p.get('frames', 1) >= 3]
        if self.last_people:
            self.person_at = time.monotonic()
        pet = self.config.get('pets', {}).get('dog')
        for item in items:
            if item['label'] == 'dog' and pet:
                item['pet'] = pet
        now = time.monotonic()
        self.stats['light'] = int(image[::16, ::16].mean())         # 0..255: a dark room makes the camera useless
        self.stats.update(frames=self.stats['frames'] + 1, detect_ms=round((t1 - t0) * 1000, 1),
                          latency_ms=round((now - meta['mono']) * 1000, 1))
        self.bus.publish('eyes.seen', {
            'mono': meta['mono'], 't': time.time() - (now - meta['mono']), 'seq': meta['seq'], 'mode': self.mode(),
            'items': items, 'stats': self.stats,
            'faces': [{'name': f['name'], 'similarity': f['similarity'], 'box': f['box']} for f in self.last_faces]
            if now - self.last_face < .5 else [],
        })

    def maybe_survey(self, frame):
        """Now and then, in good light, name the small things in view and what they lie on."""
        mode = self.mode()
        period = SURVEY_S.get(mode)
        now = time.monotonic()
        if (period is None or self.survey_busy or now - self.survey_at < period
                or self.stats.get('light', 0) < SURVEY_MIN_LIGHT or self.storage == 'critical'):
            return
        self.survey_at, self.survey_busy = now, True
        image, meta = frame
        pose = interpolate(self.poses, meta['mono'])
        self.survey_pool.submit(self._survey, image.copy(), pose, meta['mono'])

    def _survey(self, image, pose, mono):
        try:
            if self.survey is None:
                import json as jsonmod
                from .detect import Detector
                names = jsonmod.loads((MODELS / 'yoloe-26n-household-seg.names.json').read_text())
                self.survey = Detector(self.cv2, MODELS / 'yoloe-26n-household-seg.onnx', threads=2,
                                       names={int(k): v for k, v in names.items()})
            items = self.survey(image)
            h, w = image.shape[:2]
            supports = [i for i in items if i['label'] in SUPPORTS]
            for item in supports:
                self._locate(item, pose, image.shape)
            things = []
            for item in items:
                if item['label'] not in SMALL_THINGS or item['score'] < .35:
                    continue
                x1, y1, x2, y2 = item['box']
                cx = (x1 + x2) / 2
                under = [s for s in supports if s['box'][0] <= cx <= s['box'][2] and s['box'][1] - 10 <= y2 <= s['box'][3]
                         and s.get('map_xy')]
                found = {'label': item['label'], 'score': item['score'], 'box': [round(v) for v in item['box']]}
                if under:
                    support = max(under, key=lambda s: (s['box'][2] - s['box'][0]) * (s['box'][3] - s['box'][1]))
                    found.update(on=support['label'], place=SUPPORTS[support['label']], map_xy=support['map_xy'])
                else:
                    self._locate(item, pose, image.shape)
                    if item.get('map_xy') and y2 > h * .55:
                        found.update(place='на полу', map_xy=item['map_xy'])
                if found.get('map_xy'):
                    things.append(found)
            self.bus.publish('eyes.things', {'mono': mono, 'items': things, 'all': len(items)})
        except Exception as error:
            self.bus.publish('eyes.error', {'error': f'survey:{type(error).__name__}:{error}'[:160]})
        finally:
            self.survey_busy = False

    def resting(self):
        """On the dock and nobody talking to us: no faces, no gestures."""
        return bool(self.body.get('docked')) and self.mode() != 'focus'

    def maybe_gesture(self, frame):
        """Someone in view: read the body (a wave, hands up) and the face (a smile, sadness) in a worker."""
        now = time.monotonic()
        if (not self.last_people or self.gesture_busy or now - self.gesture_at < GESTURE_PERIOD_S
                or self.mode() == 'rest' or self.resting() or self.gesture_errors >= GESTURE_GIVE_UP
                or self.stats.get('light', 0) < SURVEY_MIN_LIGHT):
            return
        self.gesture_at, self.gesture_busy = now, True
        image, meta = frame
        people = [p for p in self.last_people if p['box'][3] - p['box'][1] >= GESTURE_MIN_PERSON_PX]
        if not people:
            self.gesture_busy = False
            return
        faces = [dict(f) for f in self.last_faces] if now - self.last_face < .5 else []
        self.gesture_pool.submit(self._gesture, image, people, faces, meta['mono'])

    def _gesture(self, image, people, faces, mono):
        try:
            if self.pose is None:
                from .gesture import Expressions, Gestures, Pose
                self.pose = Pose(MODELS / 'pose' / 'yolo26n-pose.onnx', threads=2)
                self.gestures = Gestures()
                fer = MODELS / 'fer' / 'emotion-ferplus-8.onnx'
                self.expressions = Expressions(fer) if fer.exists() else None
            from .detect import _iou
            bodies = []
            for body in self.pose(self.cv2, image):
                match = max(people, key=lambda p: _iou(p['box'], body['box']), default=None)
                if match is not None and _iou(match['box'], body['box']) > .3:
                    body.update(track=match.get('track'), name=match.get('name'), bearing=match.get('bearing'),
                                distance=match.get('distance'))
                    bodies.append(body)
            for kind, body in self.gestures.update(bodies):
                self.bus.publish('eyes.gesture', {'kind': kind, 'who': body.get('name'), 'track': body.get('track'),
                                                  'bearing': body.get('bearing'), 'distance': body.get('distance'),
                                                  'mono': mono})
            now = time.monotonic()
            if self.expressions is not None and faces and now - self.expression_at >= EXPRESSION_PERIOD_S:
                self.expression_at = now
                for face in faces:
                    if face['box'][2] - face['box'][0] < EXPRESSION_MIN_FACE_PX:
                        continue
                    probs = self.expressions(self.cv2, image, face['box'])
                    kind = self.expressions.update(face.get('name'), probs) if probs else None
                    if kind:
                        self.bus.publish('eyes.gesture', {
                            'kind': kind, 'who': face.get('name'), 'mono': mono,
                            'probs': {k: probs[k] for k in ('happiness', 'sadness', 'surprise', 'neutral')}})
            self.gesture_errors = 0
        except Exception as error:
            self.gesture_errors += 1
            self.bus.publish('eyes.error', {'error': f'gesture:{type(error).__name__}:{error}'[:160]})
        finally:
            self.gesture_busy = False

    def _assign_faces(self, people):
        if time.monotonic() - self.last_face > .6:
            return
        for face in self.last_faces:
            cx, cy = (face['box'][0] + face['box'][2]) / 2, (face['box'][1] + face['box'][3]) / 2
            owners = [p for p in people if p['box'][0] <= cx <= p['box'][2] and p['box'][1] <= cy <= p['box'][3]]
            if len(owners) == 1:
                owners[0]['face'] = True
                if face['name']:
                    owners[0]['name'] = face['name']
                else:
                    owners[0]['face_unknown'] = True
                    owners[0]['face_seen'] = face           # for learning a new angle (frame())
                # Looking at us: face roughly frontal and reasonably large.
                owners[0]['facing'] = (face['box'][2] - face['box'][0]) > 28

    def _locate(self, item, pose, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = item['box']
        item['bearing'] = round(self.geometry.bearing((x1 + x2) / 2, (y1 + y2) / 2), 4)
        distance = None
        if y2 < h * .97 and item['label'] in ('person', 'dog', 'cat', 'chair', 'couch', 'bed', 'dining table',
                                               'potted plant', 'suitcase', 'backpack', 'teddy bear', 'sports ball'):
            floor = self.geometry.floor((x1 + x2) / 2, y2)
            if floor is not None:
                distance = math.hypot(*floor)
        if y2 < h * .97 and distance is None and item['label'] in ('table', 'desk', 'armchair', 'shelf', 'cabinet',
                                                                    'drawer', 'sink'):
            floor = self.geometry.floor((x1 + x2) / 2, y2)
            if floor is not None:
                distance = math.hypot(*floor)
        if distance is None and item['label'] == 'person' and y2 - y1 > 40:
            distance = 502. * (1.65 if y1 > 2 else 0.9) / (y2 - y1)    # whole body / cut-off torso
        if distance is not None:
            item['distance'] = round(min(distance, 8.), 2)
            if pose is not None:
                xy = compose(pose, (distance * math.cos(item['bearing']), distance * math.sin(item['bearing']), 0.))
                item['map_xy'] = [round(xy[0], 2), round(xy[1], 2)]

    def _requests(self):
        while self.requests:
            topic, data = self.requests.popleft()
            if topic == 'eyes.look':
                self._look(data)
            elif topic == 'eyes.enroll':
                recent = [e for t, e in self.unknown if time.monotonic() - t < 30.]
                if recent and data.get('name'):
                    self.faces.enroll(data['name'], recent[-6:])
                    self.unknown.clear()
                self.bus.publish('eyes.enrolled', {'name': data.get('name'), 'samples': len(recent[-6:])})

    def _look(self, data):
        if self.frame is None:
            self.bus.publish('eyes.looked', {'id': data.get('id'), 'error': 'no_frame'})
            return
        image, meta = self.frame
        if self.wide is None:
            from .detect import Detector
            self.wide = Detector(self.cv2, MODELS / 'yolo26n.onnx', threads=3)
        items = self.wide(image)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        path = RUN_DIR / f"look-{data.get('id', 'x')}.jpg"
        self.cv2.imwrite(str(path), image, [self.cv2.IMWRITE_JPEG_QUALITY, 85])
        if data.get('keep'):
            album = STATE / 'album'
            album.mkdir(parents=True, exist_ok=True)
            kept = album / f"{int(time.time())}-{data.get('id', 'x')}.jpg"
            self.cv2.imwrite(str(kept), image, [self.cv2.IMWRITE_JPEG_QUALITY, 90])
            path = kept
        pose = interpolate(self.poses, meta['mono'])
        for item in items:
            self._locate(item, pose, image.shape)
        self.bus.publish('eyes.looked', {'id': data.get('id'), 'path': str(path), 'items': items,
                                         'age': round(time.monotonic() - meta['mono'], 2)})


TRIPS = Path('/proc/1/root/data/media/0/Android/data/ai.robotbrain.face/files/trips')
TRIPS_CAP = 2_000_000_000
TRIPS_CAP_LOW = 500_000_000


def prune_trips(cap=TRIPS_CAP):
    """Keep at most 2 GB of trip videos on the phone (the PC normally takes them sooner)."""
    try:
        files = sorted(TRIPS.glob('trip-*.mp4'), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > cap:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    config = load(args.config)
    pin('eyes', config)
    eyes = Eyes(config)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: eyes.stopping.set())
    eyes.run()


if __name__ == '__main__':
    main()
