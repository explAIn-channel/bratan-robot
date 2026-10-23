"""Hearing the house: what the sounds around are, not only words addressed to the robot.

Until 08.10 the robot heard only its name; the mind was told it cannot tell room sounds apart. Now YAMNet
(Google, AudioSet, 521 classes; ONNX, ~20 ms a second of sound on the phone) names them: Taffa barking,
the doorbell, a knock, a door slammed, music, laughter, people talking, the TV, a sneeze...
Only categories (never words or recordings) leave this module; the robot's own voice is never classified.

A sudden loud sound after quiet is a startle (a reflex before any thinking, as in animals and in social
robots' «looming» reflexes); the same sound again and again is habituated (published once per its gap).
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

RATE = 16000
WINDOW = 15600               # 0.975 s: one YAMNet patch
HOP_S = .5
GATE_FLOOR = .004            # quieter than this is silence: nothing to classify
STARTLE_LEVEL = .12          # rms of a 20 ms frame
STARTLE_JUMP = 8.            # ... at least this many times the loudest frame of the last second
STARTLE_GAP_S = 20.

# kind: (Russian words for the mind, AudioSet display names, threshold, urgent, gap s)
KINDS = {
    'dog': ('лает собака', ('Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper (dog)'), .35, True, 60.),
    'doorbell': ('звонок в дверь', ('Doorbell', 'Ding-dong'), .3, True, 20.),
    'knock': ('стук', ('Knock',), .35, True, 20.),
    'door': ('хлопнула дверь', ('Door', 'Slam', 'Cupboard open or close', 'Drawer open or close'), .4, False, 60.),
    'laughter': ('смех', ('Laughter', 'Giggle', 'Chuckle, chortle', 'Belly laugh', 'Snicker', 'Baby laughter'),
                 .35, True, 120.),
    'music': ('музыка', ('Music', 'Singing', 'Musical instrument', 'Pop music', 'Background music'), .45, True, 300.),
    'speech': ('рядом разговаривают', ('Speech', 'Conversation', 'Narration, monologue', 'Child speech, kid speaking'),
               .6, False, 120.),
    'tv': ('работает телевизор или радио', ('Television', 'Radio'), .4, False, 600.),
    'clapping': ('хлопают в ладоши', ('Clapping', 'Applause'), .4, True, 60.),
    'whistle': ('свистят', ('Whistling', 'Whistle'), .4, True, 60.),
    'alarm': ('звенит будильник или телефон', ('Alarm', 'Alarm clock', 'Telephone bell ringing', 'Ringtone',
                                               'Smoke detector, smoke alarm'), .4, True, 120.),
    'vacuum': ('гудит пылесос', ('Vacuum cleaner',), .5, False, 600.),
    'water': ('шумит вода', ('Water tap, faucet', 'Sink (filling or washing)'), .5, False, 300.),
    'kitchen': ('на кухне что-то готовят', ('Dishes, pots, and pans', 'Frying (food)', 'Microwave oven', 'Blender'),
                .45, False, 300.),
    'footsteps': ('слышны шаги', ('Walk, footsteps',), .5, False, 120.),
    'crying': ('кто-то плачет', ('Crying, sobbing', 'Baby cry, infant cry'), .5, True, 120.),
    'sneeze': ('кто-то чихнул', ('Sneeze',), .45, True, 60.),
    'bang': ('что-то громко грохнуло', ('Bang', 'Slap, smack', 'Thump, thud', 'Breaking', 'Shatter'), .45, True, 30.),
}


def rms(samples):
    return float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.


class Sounds:
    def __init__(self, model_dir: Path, publish, threads=1):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(model_dir / 'yamnet.onnx'), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        with open(model_dir / 'yamnet_class_map.csv', encoding='utf-8') as handle:
            names = [row['display_name'] for row in csv.DictReader(handle)]
        index = {name: i for i, name in enumerate(names)}
        self.kinds = {kind: (words, [index[n] for n in labels if n in index], threshold, urgent, gap)
                      for kind, (words, labels, threshold, urgent, gap) in KINDS.items()}
        self.publish = publish
        self.buffer = np.zeros(WINDOW, dtype=np.float32)
        self.levels = deque(maxlen=50)          # last second of 20 ms frame levels
        self.next_at = 0.
        self.last = {}                           # kind -> monotonic time published
        self.startled = -1e9
        self.windows: queue.Queue = queue.Queue(maxsize=1)
        self.stats = {'windows': 0, 'ms': None, 'dropped': 0}
        threading.Thread(target=self._run, name='sounds', daemon=True).start()

    def feed(self, mono, frame, *, noise=.002, own=False, talking=False):
        """One 20 ms frame. `own`: our speaker is playing (never classify ourselves);
        `talking`: a conversation with the robot is on (its speech is not news)."""
        x = np.frombuffer(frame, dtype='<i2').astype(np.float32) / 32768.
        n = len(x)
        self.buffer[:-n] = self.buffer[n:]
        self.buffer[-n:] = x
        level = rms(x)
        before = max(self.levels) if self.levels else 0.
        self.levels.append(0. if own else level)
        if own:
            self.next_at = mono + .7             # our own sound must leave the window first
            return
        if (level >= max(STARTLE_LEVEL, 30 * noise) and level >= STARTLE_JUMP * max(before, 1e-4)
                and mono - self.startled > STARTLE_GAP_S):
            self.startled = mono
            self.publish({'kind': 'startle', 'words': 'резкий громкий звук', 'urgent': True,
                          'level': round(level, 3)})
        if mono < self.next_at:
            return
        self.next_at = mono + HOP_S
        if max(self.levels) < max(GATE_FLOOR, 2.5 * noise):
            return                               # quiet room: nothing worth a classification
        try:
            self.windows.put_nowait((mono, self.buffer.copy(), talking))
        except queue.Full:
            self.stats['dropped'] += 1

    def _run(self):
        while True:
            mono, window, talking = self.windows.get()
            started = time.monotonic()
            try:
                scores = self.session.run(None, {'waveform': window})[0].max(axis=0)
            except Exception as error:           # never take the ears down with it
                self.stats['error'] = f'{type(error).__name__}:{error}'[:120]
                continue
            self.stats['windows'] += 1
            self.stats['ms'] = round((time.monotonic() - started) * 1000, 1)
            heard = []
            for kind, (words, indices, threshold, urgent, gap) in self.kinds.items():
                if not indices or (kind == 'speech' and talking):
                    continue
                score = float(scores[indices].max())
                if score < threshold:
                    continue
                if mono - self.last.get(kind, -1e9) < gap:
                    continue                     # habituation: the same sound again is no news
                self.last[kind] = mono
                heard.append({'kind': kind, 'words': words, 'urgent': urgent, 'score': round(score, 2)})
            for item in heard:
                self.publish(item)
