"""Voice process: ears (wake/stop), live conversation, own speech.

When the cloud voice is down (no credit, bad key, no network) the robot talks
offline: Vosk hears the whole phrase, the mind answers, Piper speaks it.

Bus in:  mind.context.state {text}   what the mind knows now (live instructions)
         voice.say {id, text, live}   speak (live=True: open a conversation with it)
         voice.answer {id, text, silent}  the mind's answer to a delegated request
         voice.end {}                  end the conversation now
Bus out: voice.state (retained) {live, speaking, ...}
         voice.event {kind: wake|false_wake|started|closed|stop|failed|..., ...}
         voice.ask {id, text, said}    delegated request for the mind
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import difflib
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import queue
import re
import threading
import time
import uuid

from ..common import bus as busmod
from ..common.config import MODELS, load, pin, secret
from ..common.persona import CHARACTER, LIVE_RULES
from ..face import client as face
from .audio import Audio, as_heard, rms
from .live import LiveSession
from .azure import AzureSpeech
from .hearing import Hearing
from .edge import EdgeSpeech
from .local import LocalSpeech, Phrase, request
from .wake import GATE_FLOOR, Confirmer, Listener, addressed, edit_distance, is_name

# The cloud refuses for a while (money, key): do not even try for 2 minutes.
LONG_DOWN = ('credit_balance_exhausted', 'insufficient_quota', 'invalid_api_key', 'account_deactivated')
LONG_DOWN_S = 6 * 3600.   # no credit (since 07.10): ask again in 6 h, not every 10 min
SHORT_DOWN_S = 30.
REPEAT_DOWN_MAX_S = 3600.  # failing in a row: 30 s, 30 s, 60, 120 … up to an hour between tries
FOLLOW_UP_S = 20.          # the talk stays open this long after the last words (12 s closed it after «ага»)
RESUME_FRAMES = 12         # 0.24 s of voice while we wait for the mind: the person goes on — merge it in
RESUME_WINDOW_S = 2.       # only right after the phrase (a pause between sentences), not any time later
# With the noise suppressor off (face 88) a TV or people talking across the room counted as «going on» and
# the answer was dropped again and again (08.10 20:43): a continuation is as loud as the phrase itself.
EMPTY_LIMIT = 4            # unclear bits («ага», a cough) before the talk is closed
ECHO_TAIL_S = 1.           # our sound still in the room (and the recognizer's delay) after it ended
HEAR_BEFORE_S = .8         # a follow-up goes to Microsoft from this long before its first word
EARS_BACK_S = 10.          # the ears' stream must have held this long before words go back to them (18.10: 3 s swung)
OWN_NAME_S = 8.            # the name in what we said this recently: hearing it is our own voice, not a call
BARGE_BEFORE_S = .3        # cut off by the name: the words with it are heard from this long before it was decoded
BARGE_FRAMES = 60          # …and a wake from our own line gets the last 1.2 s: the name, not our words before
PHONE_EARLY_S = .3         # woken by the phone: the ears' frames from this long before the phone's phrase began
MAIN_RING = 300            # frames (6 s) of what is heard now, kept for a wake by the phone
# A real «стоп» said in a conversation, heard by the open recognizer (not the tiny STOP grammar).
DIALOG_STOP = re.compile(r'^(стоп|стой|хватит|остановись)( пожалуйста)?$')
SPEAKING_HOLD_S = .8       # the gap between two sentences of one answer is still «speaking»
IDLE_HOLD_S = .5
DOWN_FILE = Path(os.environ.get('BRATAN_RUN', '/run/bratan')) / 'voice-down.json'
# At night he answers only a clear «Братан, …» and never asks to repeat (09.10 01:08: dictation at the PC woke him
# and he said «Не расслышал, повтори?» in the quiet hours).
NIGHT_UNCLEAR = re.compile(r'не\s+расслыш|повтори', re.I)
# The ears must not stay shut (08.10 19:17-19:25: no «Братан» heard for 8 minutes, a restart cured it).
STUCK_S = 120.             # a talk that has not moved this long is stuck (a live one ends in under a minute)
CONFIRM_STUCK_S = 30.      # the name check of one phrase lasts 8 s at most
SPEAKING_STUCK_S = 180.    # our own sound longer than this is a hung player
RENEW_GRAMMAR_S = 120.     # the name grammar is made afresh this often while the room is quiet (< 1 ms)


class EarsFrame(bytes):
    """20 ms from the body's ears with the room's hum taken out (voice/denoise.py); `.raw` — as heard: the house's
    sounds are named from it (a steady kettle or vacuum cleaner is just what the cleaner takes away)."""
    raw = None


class Voice:
    def __init__(self, config, bus_address=busmod.DEFAULT_ADDRESS):
        self.config = config
        self.audio = Audio()
        self.audio.volume = float(config.get('speaker_volume', 1.))
        self.key = secret(config['live_key'])
        self.local = LocalSpeech(self.audio, str(MODELS / 'piper' / (config.get('local_voice', 'ru_RU-irina-medium')
                                                                       + '.onnx')))
        self.edge = EdgeSpeech(self.audio, voice=config.get('backup_voice', 'ru-RU-DmitryNeural'))
        try:
            self.azure = AzureSpeech(self.audio, secret(config['azure_speech_key']),
                                     secret(config['azure_speech_region']),
                                     voice=config.get('backup_voice', 'ru-RU-DmitryNeural'))
        except OSError:
            self.azure = None             # no key: the free Edge voice is the first backup
        try:
            self.hearing = Hearing(secret(config['azure_speech_key']), secret(config['azure_speech_region']))
            # The dog's name is not in the service's dictionary: «дафа» → «таффа» (C52).
            self.hearing.names = [n for n in (config.get('pets') or {}).values() if n]
        except OSError:
            self.hearing = None           # no key: Vosk alone hears the offline conversation
        self.over_me = None               # Microsoft over the microphone while I speak: «стоп» through my voice (C52)
        self.over_me_cues = queue.Queue()
        if self.hearing is not None and (config.get('ears') or {}).get('over_me', True):
            from .overme import OverMe
            self.over_me = OverMe(self.hearing.key, self.hearing.host.split('.')[0], self.own_words,
                                  lambda found, text, start: self.over_me_cues.put((found, text, start)), self.event)
        self.utterance = None             # the phrase after the name, streamed to Microsoft's recognition
        self.down_until = 0.              # monotonic: until then the cloud voice is not tried
        self.down_code = None
        self.recall_down()
        self.dialog = None                # offline conversation in progress
        # 10 s of (monotonic, frame, speech probability): a reply that began while the name was being decided, and
        # what a person said over me before I stopped (19.10: 3 s lost «а сколько сейчас времени»)
        self.recent_frames = deque(maxlen=500)
        # Is someone speaking (voice/vad.py, 18.10): the gates of the name, of the phrase's end and of what goes to
        # Microsoft follow a speech detector, not loudness — the body's ears hear the room 12-20 dB below a voice.
        self.vad, self.speech, self.vad_reset = None, None, False
        try:
            from .vad import Silero
            self.vad = Silero(MODELS / 'silero' / 'silero_vad_v6.onnx')
        except Exception as error:          # no model: the loudness gates as before
            print('vad_unavailable', type(error).__name__, error, flush=True)
        self.spoken_log = deque(maxlen=60)       # (monotonic, text) we said, to tell our echo from a person
        self.speech_generation = 0    # 0.6 s of microphone, to hear an answer that starts at once
        self.gaps = 0                 # microphone gaps (the frame queue overflowed: we fell behind)
        self.last_mono = 0.
        self.levels = deque(maxlen=125)
        self.level_now = 0.
        self.context = ''
        self.session: LiveSession | None = None
        self.confirmer = None
        self.wake_at = 0.
        self.say_queue: queue.Queue = queue.Queue()
        self.warm = None                  # (connected LiveSession, monotonic) kept ready while someone is near
        self.bus = busmod.Client('voice', ('mind.context.state', 'voice.say', 'voice.answer', 'voice.end',
                                           'voice.prewarm', 'voice.sfx', 'voice.listen', 'voice.ab', 'voice.locate'),
                                 self.on_message, bus_address)
        self.stopping = threading.Event()
        self.timings = {}
        # The face and the mood light follow the conversation (owner, 05.10: listening and speaking must be
        # visibly different colours): listening blue, thinking violet, speaking green, then back to the mood.
        self.activity = 'idle'
        self.activity_wanted = ('idle', 0.)
        self.activity_seq = 0
        self.face_queue: queue.Queue = queue.Queue(maxsize=16)
        self.sounds = None
        try:
            from .sounds import Sounds
            self.sounds = Sounds(MODELS / 'yamnet', self.sound_heard)
        except Exception as error:          # no model: the robot just does not hear the house
            print('sounds_unavailable', type(error).__name__, error, flush=True)
        # The ears in the body (17.10, voice/array.py): where a sound came from; the words too when ears.asr allows.
        self.ears = self.direction = None
        self.ears_source, self.ears_switches = 'phone', 0
        ears = config.get('ears') or {}
        if ears.get('enabled'):
            try:
                from .array import ArrayLink, Calibration, Direction
                # Voice from the three microphones' stream by default (C52): half the Wi-Fi traffic of a separate
                # voice stream; 'a' — the board's own pick as a second stream, as before.
                from .array import LOCAL_PORT
                self.ears = ArrayLink(ears.get('host'), voice=ears.get('voice', 'multi'), on_voice=self.ears_frame,
                                      local_port=LOCAL_PORT if ears.get('bluetooth', True) else None,
                                      mix=ears.get('voice_mix', 'best'), null=self.robot_null(ears))
                self.ears.robot = self.robot_sound
                self.direction = Direction(self.ears.ring, Calibration(ears.get('calibration')))
            except Exception as error:      # no numpy/audioop or a bad config: the phone's microphone alone
                print('ears_unavailable', type(error).__name__, error, flush=True)
                self.ears = self.direction = None
        # The ears' room hum taken out before words are heard (C52, voice/denoise.py: +11 dB of voice over the hum,
        # Microsoft's words 75→75 / 84→86 / 86→88 % at 4 / 2 / 1 m); the house's sounds are named from the raw frames.
        self.denoise = None
        if self.ears is not None and ears.get('denoise', True):
            from .denoise import Denoiser
            self.denoise = Denoiser()
        self.audio.gate = lambda: self.listen_with() == 'phone'
        self.ab = None
        self.audio.tap = self.phone_tap
        # The name by the phone's microphone too while the ears hear words (18.10, the owner: «далеко не с первого раза
        # откликнулся» — one grammar on one microphone missed his «Братан» again and again): whichever hears it first
        # wakes me; the talk itself goes on by the ears (phone_names).
        self.phone_listener = None
        self.phone_frames = queue.Queue(maxsize=250)
        self.main_ring = deque(maxlen=MAIN_RING)

    # ------------------------------------------------------------ ears in the body
    @staticmethod
    def robot_null(ears):
        """My own voice taken out of the ears while I speak (C52, voice/nullvoice.py); its pattern kept in the state
        folder. ears.null = false — off; only with the voice made from the three microphones."""
        if not ears.get('null', True) or ears.get('voice', 'multi') != 'multi' or ears.get('voice_mix', 'best') == 'sum':
            return None
        from ..common.config import ASSETS, STATE
        from .nullvoice import RobotNull
        return RobotNull(ears.get('null_file') or STATE / 'robot-voice-null.npz', seed=ASSETS / 'robot-voice-null.npz')

    def robot_sound(self):
        """For the ears' null (the ears' thread): (learn, clean) — I am speaking now: learn my voice's pattern; my sound
        may still be in the room (speaking, or less than ECHO_TAIL_S since it ended or was cut off): take it out."""
        audio = self.audio
        speaking = audio.speaking
        ended = max(audio.speaker_until, getattr(audio, 'stopped_at', 0.))
        return speaking, speaking or time.monotonic() - ended < ECHO_TAIL_S

    def listen_with(self, now=None):
        """Whose microphone hears words now: ears.asr = phone | auto | array. While we speak (and our echo dies
        away) the phone listens — it cancels our own voice — unless the ears can take it out themselves (their null
        is learned and deep, voice/nullvoice.py: the owner's «подожди, а сколько времени?» over my story was heard
        there, not on the phone)."""
        mode = (self.config.get('ears') or {}).get('asr', 'phone')
        source = 'phone'
        if self.ears is not None and mode in ('auto', 'array'):
            now = time.monotonic() if now is None else now
            ours = (self.audio.speaking or now - self.audio.speaker_until < ECHO_TAIL_S) and not (
                getattr(self.ears, 'null_good', None) and self.ears.null_good())
            # Back to the ears only once their stream has held for EARS_BACK_S (18.10, Wi-Fi at −91 dBm: six switches
            # in a minute, each one restarting the speech detector mid-word); away from them at once.
            healthy = self.ears.healthy(now)
            since = getattr(self, 'ears_healthy_since', -1e9)
            if not healthy:
                since = None
            elif since is None:
                since = now
            self.ears_healthy_since = since
            steady = healthy and now - since >= EARS_BACK_S
            if self.ears_source == 'array' and not steady:
                # Hearing by the ears already: keep them while they are merely lossy (18.10 18:45, 12–25 % lost on a
                # crowded evening radio: the source swung ears ↔ phone every few seconds, each swing restarting the
                # recognizers mid-word — «не с первого раза откликнулся»). Away only when they really fail.
                steady = self.ears.usable(now)
            if steady and not (mode == 'auto' and ours):
                source = 'array'
        if source != self.ears_source:
            self.ears_source = source
            self.ears_switches += 1
            self.vad_reset = True             # another microphone: the speech detector starts afresh
        return source

    def phone_tap(self, mono, frame):
        """Every frame of the phone's microphone (the audio thread): the A/B, and the second name listener while the
        ears are the ones heard (handed to the main loop: the grammar is fed there only)."""
        self.ab_tap('phone', mono, frame)
        if self.phone_listener is not None and frame is not None and self.ears_source == 'array':
            try:
                self.phone_frames.put_nowait((mono, frame))
            except queue.Full:
                pass

    def phone_names(self, session):
        """Feed the phone's frames to its name grammar; True if it heard the name and I woke by it."""
        free = session is None and self.dialog is None and self.confirmer is None and not self.audio.speaking
        while True:
            try:
                mono, frame = self.phone_frames.get_nowait()
            except queue.Empty:
                return False
            if self.ears_source != 'array':
                continue                                      # the phone is heard anyway: the main grammar has it
            if self.audio.speaking:
                # The ears hear through my voice now (their null); the phone hears mostly me — its grammar rests and
                # starts afresh after (it would only decode my own words).
                if not getattr(self, 'phone_resting', False):
                    self.phone_resting = True
                    self.phone_listener.reset()
                    self.phone_listener.preroll.clear()
                continue
            self.phone_resting = False
            heard = self.phone_listener.feed(mono, frame, listen_for_name=free, speech=None)
            finals = self.phone_listener.finals
            if finals and finals[-1] is not getattr(self, 'phone_final_seen', None):
                self.phone_final_seen = finals[-1]
                self.event('grammar_heard', text=finals[-1][1], source='phone_second')
            if heard == 'wake' and free:
                # Only the alarm comes from the phone: the talk is heard by the ears that were heard already, from the
                # same moment on (18.10 21:11: switching the talk to the phone restarted the speech detector mid-word,
                # the phrase was taken as ended 1.5 s after the name and «…про космос» was lost).
                since = mono - len(self.phone_listener.phrase) * .02 - PHONE_EARLY_S
                frames = [f for t, f in self.main_ring if t >= since] or list(self.phone_listener.phrase)
                self.event('wake_by_phone', word=getattr(self.phone_listener, 'woke_by', ''))
                self.on_wake(frames)
                return True

    def ears_frame(self, mono, frame):
        self.ab_tap('array', mono, frame)
        heard = frame
        if frame is not None and self.denoise is not None:
            try:
                heard = EarsFrame(self.denoise.process(frame))       # every frame: the hum estimate stays fresh
                heard.raw = frame
            except Exception as error:                               # never take the ears down: raw frames then
                self.denoise = None
                self.event('denoise_failed', error=f'{type(error).__name__}: {error}'[:160])
                heard = frame
        elif frame is None and self.denoise is not None:
            self.denoise.reset()                                      # a gap: start the hum estimate afresh
        if self.listen_with() == 'array':
            self.audio.offer(mono, heard)

    # A/B (C51): the same speech through the phone's microphone and the body's array at once, two WAV files on one
    # time line — tests/phone_ears_ab.py sends both to Microsoft's recognition and compares (owner: decide by numbers).
    AB_MAX_S = 180.

    def ab_start(self, seconds, name=''):
        seconds = max(3., min(self.AB_MAX_S, seconds))
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.ab = {'start': time.monotonic(), 'until': time.monotonic() + seconds, 'phone': [], 'array': [],
                   'multi': [], 'played': [], 'name': re.sub(r'[^\w-]', '', name) or stamp}
        if self.ears is not None:
            # The three microphones as well (C52): is a sum of three, aligned by the delays, heard better?
            self.ears.on_multi = self.ab_multi
        # And what we said ourselves, when (C52): to take our own voice out of the ears.
        self.audio.played_tap = self.ab_played
        self.event('ab_started', seconds=seconds, name=self.ab['name'])

    def ab_tap(self, source, mono, frame):
        ab = getattr(self, 'ab', None)
        if ab is not None and frame is not None and ab['start'] <= mono <= ab['until']:
            ab[source].append((mono, frame))

    def ab_played(self, start, pcm):
        ab = getattr(self, 'ab', None)
        if ab is not None and ab['start'] - 5. <= start <= ab['until']:
            ab['played'].append((start, pcm))

    def ab_multi(self, epoch, first, block, mono):
        ab = getattr(self, 'ab', None)
        if ab is not None and ab['start'] - .5 <= mono <= ab['until'] + .5:
            ab['multi'].append((epoch, first, block.copy(), mono))

    def ab_finish(self, now):
        ab = getattr(self, 'ab', None)
        if ab is None or now < ab['until'] + .5:
            return
        self.ab = None
        if self.ears is not None:
            self.ears.on_multi = None
        self.audio.played_tap = None
        threading.Thread(target=self._ab_write, args=(ab,), name='ab-write', daemon=True).start()

    def _ab_write(self, ab):
        import wave
        from ..common.config import STATE
        folder = STATE / 'ab'
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for source in ('phone', 'array'):
            track = ab_track(ab[source], ab['start'], ab['until'])
            path = folder / f'{ab["name"]}-{source}.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(track)
            paths[source] = str(path)
        if ab.get('played'):
            # Our own voice on the same time line (as queued to the speaker; it sounds a little later).
            track = bytearray(int(round((ab['until'] - ab['start']) * 16000)) * 2)
            for start, pcm in ab['played']:
                at = int(round((start - ab['start']) * 16000)) * 2
                piece = pcm[max(0, -at):]
                at = max(0, at)
                piece = piece[:max(0, len(track) - at)]
                track[at:at + len(piece)] = piece
            path = folder / f'{ab["name"]}-played.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(bytes(track))
            paths['played'] = str(path)
        multi = ab_multi_track(ab.get('multi') or [], ab['start'])
        if multi is not None:
            import numpy as np
            path = folder / f'{ab["name"]}-multi.npz'
            np.savez_compressed(path, **multi)
            paths['multi'] = str(path)
        self.event('ab_saved', name=ab['name'], paths=paths,
                   frames={s: len(ab[s]) for s in ('phone', 'array', 'multi') if s in ab},
                   seconds=round(ab['until'] - ab['start'], 1))

    def heard_from(self, seconds=2.):
        """Where the sound of the last seconds came from (the array): fields for a voice event, {} if unknown."""
        if self.direction is None:
            return {}
        try:
            # Our own voice comes from the front (the phone's speaker by М1): its seconds do not count.
            found = self.direction.estimate(seconds, not_before=self.audio.speaker_until + .3)
        except Exception as error:
            return {'bearing_error': f'{type(error).__name__}: {error}'[:120]}
        if not found:
            return {}
        return {'bearing': round(found['bearing'], 3), 'bearing_conf': found['confidence'], 'side': found['side'],
                'bearing_by': found['method'], 'loudest': found['loudest'], 'excess': found['excess'],
                'tau23': found['tau23']}

    def sound_heard(self, item):
        # The sound's own «kind» (dog, music...) travels as «sound»: «kind» is the voice event's type.
        data = {('sound' if key == 'kind' else key): value for key, value in item.items()}
        data.update(self.heard_from(1.))
        self.event('sound', **data)

    # ------------------------------------------------------------ bus
    def on_message(self, topic, data):
        if topic == 'mind.context.state':
            self.context = data.get('text', '')
        elif topic == 'voice.say':
            self.say_queue.put(data)
        elif topic == 'voice.sfx':
            self.say_queue.put({'sfx': str(data.get('emotion') or '')})
        elif topic == 'voice.listen':
            # The button on his back pressed twice: as if called by name — «Да?» and listen.
            if self.dialog is None and self.session is None and self.confirmer is None:
                self.wake_at = time.monotonic()
                self.event('wake', by='button')
                self.start_dialog('')
        elif topic == 'voice.answer_part':
            dialog = self.dialog
            if (dialog is not None and data.get('id') == dialog.get('id') and data.get('text')
                    and self.quiet() and NIGHT_UNCLEAR.search(data['text']) and not self.talk_going(dialog)):
                dialog['night_unclear'] = True          # not said at night; the talk ends with the answer
                return
            if dialog is not None and data.get('id') == dialog.get('id') and data.get('text'):
                if not dialog.get('first_part_at'):
                    dialog['first_part_at'] = time.monotonic()
                    self.event('local_answer_started', after_heard=round(time.monotonic() - dialog['asked_at'], 2))
                self.say_queue.put({'text': data['text'], 'local': True})
            return
        elif topic == 'voice.answer':
            dialog = self.dialog
            if str(data.get('id') or '').startswith('local-'):
                self.event('answer_in', id=data.get('id'), mine=dialog.get('id') if dialog else None,
                           silent=bool(data.get('silent')))
            if dialog is not None and data.get('id') == dialog.get('id'):
                if dialog.get('night_unclear') or (self.quiet() and NIGHT_UNCLEAR.search(data.get('text') or '')
                                                   and not self.talk_going(dialog)):
                    self.event('night_unclear_dropped', text=(data.get('text') or '')[:120])
                    self.end_dialog('night_unclear')
                    return
                dialog['waiting'] = False
                dialog['pending'] = False
                dialog['first_part_at'] = None
                text = (data.get('text') or '').strip()
                if text and not data.get('silent'):
                    dialog['said'].append(text)
                    if data.get('streamed'):
                        self.say_queue.put({'dialog_end': True})      # the parts are queued already
                    else:
                        self.say_queue.put({'text': text, 'local': True, 'dialog': True})
                else:
                    dialog['await_quiet'] = True
                return
            if data.get('streamed'):
                return          # its parts were spoken already (07.10: the whole reply came again)
            session = self.session
            if session is not None and session.active:
                session.answer(data.get('id'), data.get('text', ''), silent=bool(data.get('silent')))
            elif data.get('text') and not data.get('silent'):
                self.say_queue.put({'text': data['text']})
        elif topic == 'voice.prewarm':
            self.prewarm()
        elif topic == 'voice.ab':
            self.ab_start(float(data.get('seconds', 30.)), str(data.get('name') or ''))
        elif topic == 'voice.locate':
            # «Ку-ку! Где ты?» (a game): where the voice of the last seconds came from — not our own call (C52).
            seconds = max(1., min(8., float(data.get('seconds', 3.))))
            threading.Thread(target=lambda: self.event('located', id=data.get('id'), **self.heard_from(seconds)),
                             name='locate', daemon=True).start()
        elif topic == 'voice.end':
            session = self.session
            if session is not None:
                session.close()
            if self.dialog is not None:
                self.end_dialog('ended')                 # the offline talk too (08.10: it could not be closed)

    def event(self, kind, **data):
        self.bus.publish('voice.event', {'kind': kind, 't': time.time(), **data})

    # ------------------------------------------------------------ face activity
    def wanted_activity(self):
        if self.audio.speaking:
            return 'speaking'
        dialog = self.dialog
        if dialog is not None:
            return 'thinking' if dialog.get('waiting') else 'listening'
        if self.confirmer is not None or getattr(self, 'night_checking', False):
            return 'listening'                       # …and while Microsoft checks the name in the quiet hours
        session = self.session
        if session is not None and session.state == 'live' and not getattr(session, 'speak_only', False):
            return 'listening'
        return 'idle'

    def activity_tick(self, now):
        wanted = self.wanted_activity()
        if wanted != self.activity_wanted[0]:
            self.activity_wanted = (wanted, now)
        if wanted == self.activity:
            return
        held = now - self.activity_wanted[1]
        if self.activity == 'speaking' and held < SPEAKING_HOLD_S:
            return
        if wanted == 'idle' and held < IDLE_HOLD_S:
            return
        was, self.activity = self.activity, wanted
        self.activity_seq += 1
        try:
            if was == 'idle':
                # Heard: the dark screen and the lamp wake at once, not when the mind sees the talk start (19.10, the
                # owner: «на базе подсветка далеко не сразу загорается, как и мордочка» — ~3 s after «Братан»).
                self.face_queue.put_nowait('/display?awake=1')
            self.face_queue.put_nowait(f'/emotion?state={wanted}&source=voice&event_id=v{self.activity_seq}')
        except queue.Full:
            pass
        self.publish_state()                             # the mind keeps the screen awake by it

    def face_loop(self):
        while not self.stopping.is_set():
            try:
                paths = [self.face_queue.get(timeout=1.)]
            except queue.Empty:
                continue
            while not self.face_queue.empty():
                try:
                    paths.append(self.face_queue.get_nowait())
                except queue.Empty:
                    break
            newest = {}                                  # only the newest of each kind matters, in order
            for path in paths:
                newest.pop(path.split('?')[0], None)
                newest[path.split('?')[0]] = path
            for path in newest.values():
                try:
                    face.post(path, 'face.write', timeout=2.)
                except Exception:
                    pass

    def publish_state(self):
        session = self.session
        listener = getattr(self, 'listener', None)
        conversing = bool(session and session.active and not getattr(session, 'speak_only', False)) \
            or self.dialog is not None
        # Why the name is (not) listened for right now (08.10 19:20: «Братан» heard by the grammar offline,
        # never acted on live).
        ears = {'session': getattr(session, 'state', None) if session is not None else None,
                'dialog': self.dialog is not None, 'confirmer': self.confirmer is not None,
                'speaking': self.audio.speaking, 'playing': bool(self.audio.status.get('playing')),
                'gaps': self.gaps, 'lag_s': round(time.monotonic() - self.last_mono, 2) if self.last_mono else None,
                'names_heard': listener.name_finals if listener else None,
                'last_heard': list(listener.finals)[-3:] if listener else None,
                'cooldown_s': round(listener.cooldown - self.last_mono, 1) if listener and self.last_mono else None,
                'named': listener.named if listener else None,
                'renewed': getattr(listener, 'renewed', 0) if listener else None,
                'speech': None if getattr(self, 'speech', None) is None else round(self.speech, 2),
                'vad_ms': None if getattr(self, 'vad_ms', None) is None else round(self.vad_ms, 2),
                'hearing': dict(self.hearing.stats) if getattr(self, 'hearing', None) else None,
                'over_me': dict(self.over_me.stats, **self.over_me.hearing.stats)
                if getattr(self, 'over_me', None) else None}
        self.bus.publish('voice.state', {'live': conversing, 'speaking': self.audio.speaking,
                                         'activity': getattr(self, 'activity', 'idle'),
                                         'audio': self.audio.mode, 't': time.time(),
                                         'noise': round(listener.noise, 5) if listener else None,
                                         'open_share': round(listener.frames_open / max(1, listener.frames_total), 3)
                                         if listener else None, 'timings': self.timings, 'ears': ears,
                                         'array': {**self.ears.summary(), 'asr': self.ears_source,
                                                   'switches': self.ears_switches} if self.ears else None})

    def instructions(self):
        return CHARACTER + '\n' + LIVE_RULES + ('\n\nЧТО ТЫ ЗНАЕШЬ О СЕБЕ СЕЙЧАС:\n' + self.context[:5000]
                                                if self.context else '')

    # ------------------------------------------------------------ sessions
    def new_session(self):
        session = LiveSession(self.key, self.audio, instructions=self.instructions, voice=self.config['live_voice'],
                              on_event=lambda kind, **d: self.session_event(session, kind, **d),
                              proxy_url=face.proxy_url)
        session.ask = lambda delegation_id, text, said: self.bus.publish(
            'voice.ask', {'id': delegation_id, 'text': text, 'said': said})
        return session

    def session_event(self, session, kind, **data):
        if kind == 'first_audio':
            if getattr(session, 'speak_only', False):
                self.event('first_audio', **data, own_line=True)
            else:
                self.timings['first_audio_after_wake'] = round(time.monotonic() - self.wake_at, 3)
                self.event('first_audio', **data, since_wake=self.timings['first_audio_after_wake'])
        elif kind in ('heard', 'said'):
            if kind == 'said' and data.get('delta'):
                self.spoken_log.append((time.monotonic(), data['delta']))
            if kind == 'heard' and getattr(session, 'speak_only', False) and not self.audio.speaking:
                session.heard_text = getattr(session, 'heard_text', '') + data.get('delta', '')
                heard = session.heard_text.strip()
                # Our own line coming back through the microphone is not an answer.
                if len(heard) >= 6 and difflib.SequenceMatcher(
                        None, heard.casefold(), session.own_text.casefold()[:len(heard) + 10]).ratio() < .5:
                    session.speak_only = False  # the person answered our line: now it is a conversation
                    session.deadline = None
                    self.publish_state()
            return
        else:
            self.event('live_' + kind, **data)
            if kind == 'failed':
                code = data.get('code') or ''
                self.down_code = code or 'failed'
                # Failing again and again (18.10: the proxy to the live voice refused every wake, 30 s apart): rest
                # longer each time — every try opened a session that the wake's own logic then had to wait out.
                self.live_failures = getattr(self, 'live_failures', 0) + 1
                pause = min(REPEAT_DOWN_MAX_S, SHORT_DOWN_S * 2 ** max(0, self.live_failures - 2))
                self.down_until = time.monotonic() + (LONG_DOWN_S if code in LONG_DOWN else pause)
                if code in LONG_DOWN:
                    self.remember_down(code, LONG_DOWN_S)
            elif kind == 'started':
                self.down_code, self.down_until = None, 0.
                self.live_failures = 0
                self.remember_down(None, 0)
        if kind in ('closed', 'aborted', 'failed', 'started'):
            if kind != 'started' and self.session is session:
                self.session = None
            self.publish_state()

    def remember_down(self, code, seconds):
        """Survive a restart: no money on the cloud voice is not forgotten by a redeploy."""
        try:
            DOWN_FILE.write_text(json.dumps({'code': code, 'until': time.time() + seconds}))
        except OSError:
            pass

    def recall_down(self):
        try:
            value = json.loads(DOWN_FILE.read_text())
        except (OSError, ValueError):
            return
        left = value.get('until', 0) - time.time()
        if left > 0 and value.get('code'):
            self.down_code, self.down_until = value['code'], time.monotonic() + left

    def live_down(self):
        return time.monotonic() < self.down_until

    def quiet(self):
        """The quiet hours (the same hours the mind sleeps in)."""
        try:
            hour = datetime.now(ZoneInfo(self.config.get('timezone', 'Europe/Moscow'))).hour
            start, end = int(self.config.get('quiet_start_hour', 23)), int(self.config.get('quiet_end_hour', 10))
        except Exception:
            return False
        if start == end:
            return True                     # equal hours: asleep all day (11.10)
        return (hour >= start or hour < end) if start > end else start <= hour < end

    def prewarm(self):
        """Someone is looking at us: open the socket now so a wake starts the session at once."""
        warm = self.warm
        if self.live_down() or self.session is not None or (warm and warm[0].state in ('connecting', 'connected')
                                        and time.monotonic() - warm[1] < 40):
            return
        if warm:
            warm[0].abort()
        session = self.new_session()
        session.connect()
        self.warm = (session, time.monotonic())

    def on_wake(self, frames):
        self.wake_at = time.monotonic()
        self.event('wake', **self.heard_from(2.))
        own = self.session
        if own is not None and getattr(own, 'speak_only', False):
            own.abort()                         # the person comes first: drop our own line
            self.session = None
        if self.live_down():
            self.event('local_mode', code=self.down_code)
            # the whole phrase is needed offline
            self.confirmer = Confirmer(self.model, list(frames), noise=self.listener.noise)
            if self.utterance is not None:
                self.utterance.cancel()
            self.utterance = self.hearing.start(list(frames)) if self.hearing else None
            return
        warm, self.warm = self.warm, None
        if warm and warm[0].state in ('connecting', 'connected') and time.monotonic() - warm[1] < 50:
            session = warm[0]
            self.event('warm_session')
        else:
            if warm:
                warm[0].abort()
            session = self.new_session()
            session.connect()
        self.session = session
        frames = list(frames)
        threading.Thread(target=self._begin, args=(session, frames), name='live-begin', daemon=True).start()
        self.confirmer = Confirmer(self.model, frames, noise=self.listener.noise)
        # The cloud voice may refuse after all (07.10 19:06: the 10-minute retry): then Microsoft's text,
        # not Vosk's, starts the offline conversation.
        if self.utterance is not None:
            self.utterance.cancel()
        self.utterance = self.hearing.start(frames) if self.hearing else None

    def _begin(self, session, frames):
        if not session.begin(frames):
            if self.session is session:
                self.session = None
            self.event('live_unavailable')
            if not session.hold and self.confirmer is None and self.dialog is None:
                # Confirmed already, the words went to the dead session: ask again offline.
                self.start_dialog('')
        else:
            self.timings['session_ready_after_wake'] = round(time.monotonic() - self.wake_at, 3)

    def confirm(self, early=False):
        if early:
            ok, text = True, 'имя в начале фразы'
        else:
            ok, text = self.confirmer.result()
        self.confirmer = None
        utterance, self.utterance = self.utterance, None
        session = self.session
        self.event('confirmed' if ok else 'false_wake', text=text,
                   decided_after_wake=round(time.monotonic() - self.wake_at, 3))
        if not ok:
            if session is not None:
                session.abort()
            self.session = None
            if utterance is not None:
                # Vosk heard no name up front; Microsoft hears better (07.10 17:19 «братан не уволен
                # должна быть тона» was dropped): if it hears the name first, the conversation starts.
                threading.Thread(target=self._second_opinion, args=(utterance,), name='hear-wake',
                                 daemon=True).start()
            return
        if session is not None and session.state in ('connecting', 'connected', 'live'):
            if utterance is not None:
                utterance.cancel()
            session.release()
        elif utterance is not None and self.quiet():
            # Night: Microsoft must hear the name first as well, before anything is said (no «Да?»); the face shows
            # it is listening meanwhile (19.10, the owner wants it to react at once).
            self.night_checking = True

            def check():
                try:
                    self.guarded(self._night_call, utterance, text)
                finally:
                    self.night_checking = False
            threading.Thread(target=check, name='hear-night', daemon=True).start()
        else:
            self.start_dialog(text, utterance)

    def _night_call(self, utterance, vosk_text):
        heard = self.heard(utterance, vosk_text)
        if heard and self.dialog is None and self.session is None and (
                addressed(heard.split()) or (addressed((vosk_text or '').split()) and sounds_like_name(heard))):
            if not addressed(heard.split()):                 # «брэдан, сколько…» → «братан, сколько…»
                words = heard.split()
                at = next(i for i, w in enumerate(words[:2]) if name_like(w))
                heard = ' '.join(words[:at] + ['братан'] + words[at + 1:])
            self.event('night_call', text=heard)
            self.start_dialog(heard)
        else:
            self.event('night_not_addressed', text=heard, vosk=vosk_text)

    # ------------------------------------------------------------ offline conversation
    def start_dialog(self, text, utterance=None):
        """The cloud voice is down: hear, ask the mind, answer with the offline voice."""
        if not self.local.ready.is_set() and not self.edge.cache and not (self.azure and self.azure.cache):
            if utterance is not None:
                utterance.cancel()
            self.event('unavailable', code=self.down_code or 'live_failed')
            return
        dialog = self.dialog = {'id': None, 'heard': [], 'said': [], 'phrase': None, 'listen_until': 0.,
                                'waiting': utterance is not None, 'await_quiet': False,
                                'asked_at': time.monotonic()}
        self.event('local_started', code=self.down_code)
        self.publish_state()

        def first(heard):
            if self.dialog is not dialog:
                return
            words = request(heard)
            if words:
                self.dialog_ask(words, vosk=request(text) if heard != text else None, named=True)
            elif self.last_mono < self.listener.open_until:
                # Still talking after the name: hear the rest, no «Да?» over it (08.10: «Как дела?» was lost
                # under our «Да?»).
                dialog['waiting'] = False
                dialog['heard_from'] = self.last_mono - .8
                dialog['await_quiet'] = True
            else:
                dialog['waiting'] = False
                dialog['said'].append('Да?')
                self.say_queue.put({'text': 'Да?', 'local': True, 'dialog': True})
        if utterance is None:
            self.guarded(first, text)
        else:
            threading.Thread(target=lambda: self.guarded(lambda: first(self.heard(utterance, text))),
                             name='hear-first', daemon=True).start()

    def _second_opinion(self, utterance):
        text = utterance.result()
        if text is None:
            return
        if text and (addressed(text.split()) or sounds_like_name(text)) and self.dialog is None and self.session is None:
            # The name grammar woke me on «Братан»; Microsoft may write it «бренд» (18.10): in front, it is the call.
            self.event('confirmed_by_azure', text=text)
            self.start_dialog(text)
        else:
            self.event('false_wake_azure', text=text)

    def heard(self, utterance, vosk_text):
        """Microsoft's text when it came in time, else Vosk's (an empty text from Microsoft: nothing said).

        18.10: in 12 days Microsoft said «nothing» 21 times for phrases Vosk made out («какая погода на улице»,
        «расскажи мне пожалуйста», «иди поищи …»): the robot took them for a cough and the words were lost. Now
        «nothing» against two or more words of Vosk's is heard once more; nothing again — Vosk's words if three or
        more (one or two are what Vosk makes of a TV or a far voice: «салют», «пока»)."""
        text = utterance.result() if utterance is not None else None
        if text is None:
            return vosk_text
        words = len(request(vosk_text or '').split())
        if text or words < 2:
            return text
        again = utterance.again()
        second = again.result(wait=4.) if again is not None else None
        chosen = second if second else (vosk_text if words >= 3 else '')
        self.event('heard_again', vosk=vosk_text, heard=second, chosen=chosen)
        return chosen

    def listen_for_answer(self, line):
        """We spoke to someone offline (07.10: «Таффа не с тобой?» — the answer was not heard): listen
        for the answer without the name, like after an answer in a conversation."""
        if self.dialog is not None or not (self.local.ready.is_set() or (self.azure and self.azure.cache)):
            return
        self.dialog = {'id': None, 'heard': [], 'said': [line], 'phrase': None, 'listen_until': 0.,
                       'waiting': False, 'await_quiet': True, 'asked_at': 0.}
        self.event('local_started', code=self.down_code, after_own_line=True)
        self.publish_state()

    def dialog_ask(self, words, vosk=None, named=False, follow_up=False):
        dialog = self.dialog
        dialog['id'] = 'local-' + uuid.uuid4().hex[:8]
        dialog['waiting'], dialog['asked_at'] = True, time.monotonic()
        dialog['pending'] = True                          # until its answer comes: what follows is merged in
        dialog['voice_level'] = self.voice_level()
        if not dialog.get('prefix_used'):
            dialog['merges'] = 0                          # a new question may be continued once
        dialog['prefix_used'] = False
        dialog['heard'].append(words)
        extra = {'vosk': vosk} if vosk is not None and vosk != words else {}
        # Where the words came from (C52): «иди ко мне» said in a talk, out of view, is found by the voice.
        self.event('local_heard', text=words, id=dialog['id'], **extra, **self.heard_from(3.))
        # 'said' means "the live voice already answered this aloud"; offline nothing was answered yet
        # (the mind keeps our earlier lines in its dialogue) — passing them made it stay silent.
        self.bus.publish('voice.ask', {'id': dialog['id'], 'text': words, 'said': '', 'named': named,
                                       'follow_up': follow_up})

    def end_dialog(self, reason):
        dialog, self.dialog = self.dialog, None
        if dialog is None:
            return
        if dialog.get('utterance') is not None:
            dialog['utterance'].cancel()
        self.event('live_closed', heard=' '.join(dialog['heard']), said=' '.join(dialog['said']),
                   local=True, reason=reason)
        self.publish_state()

    def start_hearing(self, dialog, mono):
        """Microsoft hears a follow-up from just before its first word, not the quiet before it (until 18.10 a talk
        streamed everything from the end of our own line — up to 15 s, then nothing more: a reply after a long pause
        reached the service as silence; the quiet seconds were paid for too)."""
        phrase = dialog.get('phrase')
        if dialog.get('utterance') is not None or phrase is None or not phrase.speech or self.hearing is None:
            return
        began = mono - (phrase.frames - 1 - phrase.speech_frame) * .02
        since = max(dialog.get('listen_from', 0.), began - HEAR_BEFORE_S)
        dialog['utterance'] = self.hearing.start([f for t, f, _ in self.recent_frames if t >= since - 1e-3])

    def dialog_frame(self, mono, frame):
        dialog = self.dialog
        self.recent_frames.append((mono, frame, self.speech))
        if dialog['waiting']:
            # Once the answer is being spoken, its end may come late (the model still writes its
            # thought): 15 s cut a spoken reply off and it came again whole (07.10 17:00).
            if mono - dialog['asked_at'] > (40. if dialog.get('first_part_at') else 15.):
                self.end_dialog('mind_timeout')
            elif dialog.get('pending') and not dialog.get('first_part_at') and not self.audio.speaking:
                self.watch_resume(dialog, mono, frame)
            return
        if self.audio.speaking:
            return                                  # never hear our own voice
        if dialog['await_quiet']:
            dialog['await_quiet'] = False
            # `speaking` lasts 0.3 s past the real end of our sound: a person who answers at once
            # starts inside that tail — hear it from the moment our sound really ended.
            ended = max(self.audio.speaker_until + .05, dialog.get('heard_from', 0.))
            barged = dialog.pop('barge_from', None)
            if barged is not None:
                ended = barged                      # cut off by the name: what was said with it counts
            heard_already = [(f, p) for t, f, p in self.recent_frames if t >= ended]
            dialog['phrase'] = Phrase(self.model, [f for f, _ in heard_already], noise=self.listener.noise,
                                      max_s=FOLLOW_UP_S, probs=[p for _, p in heard_already])
            if barged is not None:
                self.event('barge_listening', kept_s=round(len(heard_already) * .02, 2),
                           voiced_s=round(dialog['phrase'].voiced * .02, 2), late_s=round(mono - barged, 2))
            dialog['utterance'] = None                     # Microsoft hears it from its first word
            dialog['listen_from'] = ended
            dialog['listen_until'] = mono + FOLLOW_UP_S
            self.start_hearing(dialog, mono)
            return
        phrase = dialog['phrase']
        if phrase is None:
            return
        phrase.feed(frame, self.speech)
        if dialog.get('utterance') is not None:
            dialog['utterance'].feed(frame)
        else:
            self.start_hearing(dialog, mono)
        if phrase.ended and not phrase.speech:
            self.end_dialog('quiet')
        elif phrase.speech and phrase.ended:
            vosk = phrase.text()
            utterance, dialog['utterance'] = dialog.get('utterance'), None
            dialog['phrase'] = None
            dialog['phrase_end'] = mono - phrase.silence * .02
            dialog['waiting'], dialog['asked_at'] = True, mono

            def ask():
                heard = self.heard(utterance, vosk)
                if self.dialog is not dialog:
                    return
                if dialog.pop('strip_own', False) and heard:
                    from .overme import strip_mine            # heard from the window they cut in: my words first
                    heard = strip_mine(heard, self.own_words(30.))
                words = request(heard)
                prefix = dialog.pop('prefix', '')
                if prefix:
                    words = (prefix + ' ' + words).strip()       # the first half, heard before the pause
                    dialog['prefix_used'] = True
                if words and DIALOG_STOP.match(words.casefold().strip(' .,!?…')):
                    self.bus.publish('body.stop', {'reason': 'voice_stop'})
                    self.event('stop', text=words, via='dialog')
                    self.end_dialog('stop')
                elif words:
                    self.dialog_ask(words, vosk=request(vosk) if heard != vosk else None,
                                    named=addressed((heard or '').split()), follow_up=True)
                elif addressed((heard or vosk or '').split()):
                    # Only the name, in a talk already open (22.10: he spoke first, the owner called «Братан» three
                    # times and heard nothing back — «он меня вообще не слышит»): answer as to a call, then listen.
                    self.event('local_name_only', heard=heard, vosk=vosk)
                    dialog['said'].append('Да?')
                    dialog['listen_until'] = time.monotonic() + FOLLOW_UP_S
                    dialog['waiting'] = False                  # spoken, then heard afresh (await_quiet)
                    self.say_queue.put({'text': 'Да?', 'local': True, 'dialog': True})
                elif dialog.get('empty', 0) < EMPTY_LIMIT and time.monotonic() < dialog['listen_until']:
                    self.event('local_unclear', vosk=vosk, heard=heard)
                    # «ну…», a cough, a bark: not an answer yet — keep listening till the time is up (08.10:
                    # a hesitation closed the talk and the reply that followed was ignored).
                    dialog['empty'] = dialog.get('empty', 0) + 1
                    dialog['listen_until'] = time.monotonic() + FOLLOW_UP_S     # someone is still talking here
                    dialog['phrase'] = Phrase(self.model, (), noise=self.listener.noise, max_s=FOLLOW_UP_S)
                    dialog['utterance'] = None                 # from its first word again
                    dialog['listen_from'] = self.last_mono
                    dialog['waiting'] = False
                else:
                    self.end_dialog('not_understood')     # noise or nothing clear: the name starts anew
            threading.Thread(target=lambda: self.guarded(ask), name='hear-follow-up', daemon=True).start()
        elif not phrase.speech and mono > dialog['listen_until']:
            self.end_dialog('quiet')

    def guarded(self, fn, *args):
        """Run a hearing step; a failure is logged and the offline talk is closed cleanly instead of hanging."""
        try:
            return fn(*args)
        except Exception as error:
            self.event('hearing_error', error=f'{type(error).__name__}: {error}'[:200])
            if self.dialog is not None:
                self.end_dialog('error')

    def microphone_effects(self):
        """Android's noise suppressor on (re-asserted: a restarted face app starts without it). Off, the room's
        music and talk kept the name's gate open 95% of the time and «Братан» was never made out (08.10 20:50);
        on, the owner's «Братан» was heard 6 of 6."""
        self.effects_at = time.monotonic()
        try:
            face.post('/audio/effects?ns=1&aec=1', 'face.write')
        except Exception:
            pass                                          # an older face app: it has the suppressor on anyway

    def voice_level(self):
        """How loud the person's voice was just now (the loud quarter of the last 2.5 s), kept by the main loop.
        Read from the hearing threads: iterating the frames there raced the main loop and killed the thread
        («deque mutated during iteration»), so the question was never sent (08.10 20:59)."""
        return self.level_now

    def watch_resume(self, dialog, mono, frame):
        """The person goes on talking while the mind is still writing the answer (a pause between two sentences:
        «Да нормально всё. — Устал немного на работе.» was answered after the first and the answer talked over
        the second, 08.10 20:19): that answer is dropped and the rest is heard and asked together."""
        if dialog.get('merges', 0) >= 1 or time.monotonic() - dialog['asked_at'] > RESUME_WINDOW_S:
            return
        threshold = max(GATE_FLOOR, self.listener.noise * 2.5, .5 * dialog.get('voice_level', 0.))
        voiced = dialog.setdefault('voiced', deque(maxlen=20))
        # As loud as the phrase itself and a voice by the speech detector (a door or a clatter is loud, not a voice).
        voiced.append(rms(as_heard(frame)) > threshold and (self.speech is None or self.speech >= .5))
        if sum(voiced) < RESUME_FRAMES:                 # syllables dip between: count, not a run (08.10 20:24)
            return
        voiced.clear()
        old = dialog['id']
        dialog['id'] = None                              # its parts are not ours any more
        dialog['pending'] = False
        dialog['merges'] = dialog.get('merges', 0) + 1
        self.bus.publish('voice.cancel', {'id': old})
        dialog['prefix'] = (dialog['heard'].pop() if dialog['heard'] else '')
        start = max(mono - (RESUME_FRAMES + 15) * .02 - 1.5, dialog.get('phrase_end', 0.))
        recent = [(f, p) for t, f, p in self.recent_frames if t >= start]
        frames = [f for f, _ in recent]
        dialog['phrase'] = Phrase(self.model, frames, noise=self.listener.noise, max_s=FOLLOW_UP_S,
                                  probs=[p for _, p in recent])
        dialog['utterance'] = self.hearing.start(frames) if self.hearing else None
        dialog['listen_until'] = mono + FOLLOW_UP_S
        dialog['waiting'] = False
        self.event('local_resumed', dropped=old, prefix=dialog['prefix'])

    def speak_loop(self):
        while not self.stopping.is_set():
            try:
                item = self.say_queue.get(timeout=.5)
            except queue.Empty:
                continue
            if item.get('dialog_end'):
                if self.dialog is not None:
                    self.dialog['await_quiet'] = True           # all parts spoken: listen for the answer
                continue
            if 'sfx' in item:
                # A sound without words: never over speech, never while someone is talking to us.
                busy = self.audio.speaking or self.dialog is not None or self.confirmer is not None or \
                    (self.session is not None and self.session.state in ('connecting', 'connected', 'live'))
                if not busy:
                    from . import sfx as sfxmod
                    from ..common.config import STATE
                    sfxmod.OWN_SOUNDS = STATE / 'sounds'      # the owner's own files win over the synthesis
                    pcm = sfxmod.synth(item['sfx'])
                    if pcm:
                        self.audio.play(pcm)
                        self.event('sfx', emotion=item['sfx'])
                continue
            text = (item.get('text') or '').strip()
            if not text:
                continue
            session = self.session
            deadline = time.monotonic() + 4.
            while (session is not None and session.state in ('connecting', 'connected')
                   and time.monotonic() < deadline):
                time.sleep(.1)                  # a wake is being decided: do not take its session
                session = self.session
            if session is not None and session.state in ('connecting', 'connected'):
                continue                        # still undecided: the person comes first
            if session is not None and session.state == 'live':
                session.instruct('Мысль мозга: скажи сейчас своими словами, коротко: «' + text[:300] + '»')
                continue
            if item.get('local') or item.get('tts_only') or self.live_down():
                self.backup_say(text, item.get('id'))
                if item.get('dialog') and self.dialog is not None:
                    self.dialog['await_quiet'] = True
                elif item.get('live'):
                    self.listen_for_answer(text)
                continue
            # One voice for everything: own lines are spoken by the live voice too.
            live = bool(item.get('live'))
            warm, self.warm = self.warm, None
            if warm and warm[0].state in ('connecting', 'connected') and time.monotonic() - warm[1] < 50:
                session = warm[0]
            else:
                if warm:
                    warm[0].abort()
                session = self.new_session()
                session.connect()
            self.session = session
            session.release()
            if not live:
                session.idle_close_s = 3.
                session.speak_only = True       # our own line: not a conversation unless someone answers
                session.own_text = text
                session.deadline = time.monotonic() + 30.
            started = time.monotonic()
            if live:
                prompt = ('[мысль мозга] Ты сам решил сказать человеку рядом: «' + text[:300] +
                          '». Скажи это вслух своими словами, коротко, и дальше разговаривай, если он ответит.')
            else:
                prompt = ('[мысль мозга] Произнеси вслух дословно, своим обычным голосом, и больше ничего '
                          'не добавляй: «' + text[:300] + '»')
            if session.begin(first_text=prompt):
                self.event('spoke', text=text[:300], id=item.get('id'), via='live',
                           ready_s=round(time.monotonic() - started, 2))
                continue
            self.session = None
            self.backup_say(text, item.get('id'))       # the cloud voice failed: the backup voice at once
            if live:
                self.listen_for_answer(text)

    def own_words_said(self, words, window_s=8.):
        """Did we ourselves say these words, or words that sound like them, in the last seconds (so hearing
        them is our echo)? The tiny STOP grammar fits any sound to its few words: our own «Стою на кухне»
        came back as «стой», cut the answer off and closed the conversation (08.10 18:53)."""
        recent = [w.strip('.,!?…«»"—:;()') for at, t in self.spoken_log if time.monotonic() - at < window_s
                  for w in t.casefold().split()]
        for word in words.casefold().split():
            if word in ('братан', '[unk]') or not word:
                continue
            if any(r and (r == word or r[:3] == word[:3] or edit_distance(r, word) <= 1) for r in recent):
                return True
        return False

    def backup_say(self, text, item_id=None):
        """Microsoft's Dmitry: official Azure first, the free Edge endpoint if Azure refuses, offline Piper last."""
        self.spoken_log.append((time.monotonic(), text))
        generation = self.speech_generation
        current = lambda: self.speech_generation == generation       # «стоп» cuts the line off
        via, first = 'azure', self.azure.say(text, current) if self.azure else None
        if first is None and current():
            via, first = 'edge', self.edge.say(text, current)
        if first is None and current():
            via, first = 'local', self.local.say(text, current)
        self.event('spoke', text=text[:300], id=item_id, via=via,
                   first_sound=None if first is None else round(first, 3))

    def ears_watch(self, now):
        """Once a second: nothing may keep the ears shut. A talk that stopped moving, a name check or our own sound
        that never ended is let go; the name grammar is made afresh while the room is quiet (08.10 19:17: no
        grammar result for 8 minutes while the same audio replayed gave five wakes)."""
        dialog = self.dialog
        if dialog is not None and not self.audio.speaking:
            mark = (id(dialog), len(dialog['heard']), len(dialog['said']), dialog.get('waiting'),
                    dialog.get('await_quiet'), dialog.get('id'))
            if mark != getattr(self, 'ears_mark', None):
                self.ears_mark, self.ears_moved_at = mark, now
            elif now - self.ears_moved_at > STUCK_S:
                self.event('ears_stuck', what='dialog', heard=' '.join(dialog['heard'])[-120:],
                           waiting=bool(dialog.get('waiting')), pending=bool(dialog.get('pending')))
                self.end_dialog('stuck')
        if self.confirmer is not None and now - self.wake_at > CONFIRM_STUCK_S:
            self.event('ears_stuck', what='confirmer')
            self.confirmer = None
            utterance, self.utterance = self.utterance, None
            if utterance is not None:
                utterance.cancel()
        if self.audio.speaking:
            self.speaking_since = getattr(self, 'speaking_since', None) or now
            if now - self.speaking_since > SPEAKING_STUCK_S:
                self.event('ears_stuck', what='speaking')
                self.speaking_since = None
                self.audio.stop_playing()
        else:
            self.speaking_since = None
        listener = getattr(self, 'listener', None)
        free = (self.session is None and self.dialog is None and self.confirmer is None and not self.audio.speaking)
        if (listener is not None and free and self.last_mono >= listener.open_until
                and now - getattr(self, 'grammar_at', 0.) > RENEW_GRAMMAR_S):
            self.grammar_at = now
            listener.renew()
            if self.phone_listener is not None:
                self.phone_listener.renew()

    def cut_speech(self):
        """Silence at once: what is being synthesized, the sentences still waiting and the sound already playing."""
        self.speech_generation += 1
        while True:
            try:
                self.say_queue.get_nowait()
            except queue.Empty:
                break
        self.audio.stop_playing()

    def can_barge(self, session):
        """While we speak on our own (a thought, an offline answer — the live voice hears through itself), the name
        cuts us off: ears.barge = false turns it off."""
        return (self.audio.speaking and self.confirmer is None
                and (session is None or bool(getattr(session, 'speak_only', False)))
                and bool((self.config.get('ears') or {}).get('barge', True)))

    @staticmethod
    def talk_going(dialog):
        """I have already answered in this talk (not only «Да?»): «не расслышал, повтори» is part of it, said even in
        the quiet hours (18.10 20:07: the owner cut in with «Подожди, а сколько времени?», it was misheard, the
        «повтори» was dropped as a night noise and the talk closed — his question lost)."""
        return any(line and line != 'Да?' for line in dialog.get('said', ()))

    def own_words(self, within=90.):
        """The words I said lately, as Microsoft writes them (to tell my own «стой» from a person's). A line is logged
        when it starts and a story lasts 20 s and more: with 20 s here my own «стёр», «маленьких» at the end of a long
        story came back as someone else's and I cut myself off twice (18.10 18:24)."""
        from .hearing import normalize
        now = time.monotonic()
        return [w for at, text in list(self.spoken_log) if now - at < within for w in normalize(text).split()]

    def over_me_cue(self, found, text, start, mono, session):
        """Microsoft heard someone over my voice (OverMe). «Стоп», «хватит»… — silence, as the name grammar's
        «Братан, стоп»; anything else said to me — «братан», «подожди», «а сколько времени?» (18.10, the owner: «хочу,
        чтобы можно было перебивать любой фразой») — silence and I listen from where they cut in (`start`)."""
        from .overme import STOP
        if any(STOP.match(w) for w in found):
            self.bus.publish('body.stop', {'reason': 'voice_stop'})
            self.cut_speech()
            self.event('stop', text=text, via='over_me')
            if self.dialog is not None:
                self.end_dialog('stop')
            if session is not None and getattr(session, 'speak_only', False):
                session.abort()
                self.session = None
            return
        self.barge_in(mono, word=found[0], start=start)

    def barge_in(self, mono, word=None, start=None):
        """«Братан…» while we speak (C52, owner: «стоп» and corrections must be heard through his own speech). The
        name is never part of what we say, so we stop at once and listen, like a smart speaker on its wake word; the
        words after it are heard with us quiet. The phone cancels our voice and heard the name in the A/B (the ears
        cannot: only ~2 dB of our voice can be taken out of them — another clock, the room's long tail); the small
        grammar did not hear «стоп» under it. Only the name itself, and not if we said it ourselves just now."""
        word = word or getattr(self.listener, 'woke_by', '')
        recent = [w.strip('.,!?…«»"—:;()') for at, t in self.spoken_log if time.monotonic() - at < OWN_NAME_S
                  for w in t.casefold().split()]
        if (start is None or is_name(word or '')) and any(is_name(w) for w in recent):
            self.event('barge_ignored', word=word, why='own_name')
            return False
        self.event('barge_in', word=word, **self.heard_from(2.))
        self.cut_speech()
        dialog = self.dialog
        if dialog is not None:
            old = dialog.get('id')
            if old and (dialog.get('pending') or dialog.get('waiting')):
                self.bus.publish('voice.cancel', {'id': old})        # the rest of that answer is not wanted now
            # Cut in by Microsoft's window: their words are already said — heard from that window on, my own words in
            # front taken off the text (strip_own); by the name grammar: from just before it was decoded.
            dialog.update(id=None, pending=False, first_part_at=None, waiting=False, await_quiet=True,
                          barge_from=start if start is not None else mono - BARGE_BEFORE_S, empty=0,
                          strip_own=start is not None)
            return True
        self.on_wake(self.listener.phrase[-BARGE_FRAMES:])      # the name and what follows, not our words before
        return True

    def detect_speech(self, frame):
        """The speech detector's probability for this frame into `self.speech` (None — no detector: loudness)."""
        if self.vad is None:
            self.speech = None
            return
        if self.vad_reset:
            self.vad_reset = False
            self.vad.reset()
        started = time.perf_counter()
        try:
            self.speech = self.vad.feed(frame)
        except Exception as error:                   # never take the ears down: back to the loudness gates
            self.event('vad_failed', error=f'{type(error).__name__}: {error}'[:160])
            self.vad, self.speech = None, None
            return
        spent = time.perf_counter() - started
        self.vad_ms = spent * 1000 if getattr(self, 'vad_ms', None) is None else .99 * self.vad_ms + 10 * spent

    # ------------------------------------------------------------ main loop
    def run(self):
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(str(MODELS / 'vosk-ru'))
        self.listener = Listener(self.model)
        if self.ears is not None and (self.config.get('ears') or {}).get('phone_wake', True):
            self.phone_listener = Listener(self.model)
        self.audio.start()
        if self.ears is not None:
            self.ears.start()
        self.bus.start()
        self.microphone_effects()
        threading.Thread(target=self.speak_loop, name='speak', daemon=True).start()
        threading.Thread(target=self.face_loop, name='face-activity', daemon=True).start()
        last_state = 0.
        while not self.stopping.is_set():
            try:
                mono, frame = self.audio.frames.get(timeout=.5)
            except queue.Empty:
                self.activity_tick(time.monotonic())
                continue
            self.activity_tick(time.monotonic())
            if frame is None:
                self.gaps += 1
                self.listener.reset()
                if self.vad is not None:
                    self.vad.reset()
                continue
            self.last_mono = mono
            self.detect_speech(frame)
            self.levels.append(rms(as_heard(frame)))
            if len(self.levels) % 25 == 0 or len(self.levels) < 125:
                ordered = sorted(self.levels)
                self.level_now = ordered[int(len(ordered) * .75)]
            session = self.session
            self.main_ring.append((mono, frame))
            if self.phone_listener is not None and self.phone_names(session):
                continue
            live = session is not None and session.state == 'live'
            if self.sounds is not None:
                talking = self.dialog is not None or (session is not None and session.active
                                                      and not getattr(session, 'speak_only', False))
                self.sounds.feed(mono, as_heard(frame), noise=self.listener.noise, own=self.audio.speaking,
                                 talking=talking)
            if session is not None and session.active:
                # Our own line must not hear itself: the echo would be answered in a loop.
                if not (getattr(session, 'speak_only', False) and self.audio.speaking):
                    session.feed(frame)
            if self.dialog is not None:
                self.dialog_frame(mono, frame)
            barge = self.can_barge(session)
            if self.over_me is not None:
                if barge and self.spoken_log and time.monotonic() - self.spoken_log[-1][0] < 30.:   # words, not a beep
                    self.over_me.feed(mono, frame)
                else:
                    self.over_me.stop()
                if not self.over_me_cues.empty():
                    self.over_me_cue(*self.over_me_cues.get_nowait(), mono=mono, session=session)
                    session = self.session
            heard = self.listener.feed(mono, frame, listen_for_name=barge or (
                session is None and self.dialog is None and self.confirmer is None and not self.audio.speaking),
                speech=self.speech, strict=barge)
            finals = self.listener.finals
            if finals and finals[-1] is not getattr(self, 'final_seen', None):
                # What the name grammar made of the room (18.10: «Братан, стоп» over our story left no trace; «Братан»
                # in quiet was answered «не с первого раза» and the misses left none either) — tuned by these.
                self.final_seen = finals[-1]
                self.event('heard_over_me' if self.audio.speaking else 'grammar_heard', text=finals[-1][1],
                           source=self.ears_source, barge=barge, speech=round(self.listener.speech_peak, 2),
                           peak=round(self.listener.peak, 4), noise=round(self.listener.noise, 4))
            stop_text = getattr(self.listener, 'last_text', '') if heard == 'stop' else ''
            if heard in ('stop', 'wake') and barge and self.over_me is not None:
                # Over my own voice the grammar is not trusted (18.10: it missed the owner's «Братан, стоп» and made
                # «братан стой» out of my «Сейчас…» twice): Microsoft checks the window now and decides (over_me_cue).
                self.event('over_me_asked', text=getattr(self.listener, 'last_text', '') or heard)
                self.over_me.check_now(mono)
                heard = None
            if heard == 'stop' and not stop_text.startswith('братан'):
                recently = self.audio.speaking or time.monotonic() - self.audio.speaker_until < ECHO_TAIL_S
                if recently:
                    # The tiny STOP grammar fits our own speech to its words: «Стою» -> «стой», «…не состыковался»
                    # -> «остановись» (08.10 20:07, it stopped itself on the way home). While we speak, only
                    # «Братан, стоп» stops us.
                    self.event('stop_ignored_while_speaking', text=stop_text)
                    heard = None
                elif self.dialog is not None and not recently:
                    # Listening for a reply: the person's words go to the open recognizer, and a short reply the
                    # STOP grammar took for «стой» must not end the talk — a real «стоп» is heard there.
                    self.event('stop_left_to_dialog', text=stop_text)
                    heard = None
            if heard == 'stop':
                self.bus.publish('body.stop', {'reason': 'voice_stop'})
                self.cut_speech()
                self.event('stop', text=stop_text)
                if self.dialog is not None:
                    self.end_dialog('stop')
                if session is not None and mono - self.wake_at < 4.:
                    # «Братан, стоп!» is a command, not the start of a conversation.
                    session.abort()
                    self.session = None
                    self.confirmer = None
            elif heard == 'wake' and barge:
                if self.barge_in(mono):
                    continue
            elif heard == 'wake' and session is None:
                self.on_wake(self.listener.phrase)
                continue
            if self.confirmer is not None:
                self.confirmer.feed(frame, self.speech)
                if self.utterance is not None:
                    self.utterance.feed(frame)
                if self.session is not None and self.confirmer.early():
                    self.confirm(early=True)
                elif self.confirmer.ended:
                    self.confirm()
            if mono - last_state > 1.:
                last_state = mono
                self.publish_state()
                self.ears_watch(time.monotonic())
                self.ab_finish(time.monotonic())
                if time.monotonic() - getattr(self, 'effects_at', 0.) > 300:
                    self.microphone_effects()


def sounds_like_name(text):
    """Microsoft's first two words hold something written close to «братан» (18.10: «брэдан», «бретон» — two letters
    off): with Vosk's clear «Братан» in front it is the name, and the quiet hours' check lets the call through."""
    return any(name_like(w) for w in (text or '').split()[:2])


def name_like(word):
    """«брэдан», «бретон», «бракенд», «браке» — what Microsoft writes for «Братан» it does not know (18.10)."""
    return len(word) >= 4 and (edit_distance(word, 'братан') <= 2 or word.startswith(('бра', 'бре', 'брэ')))


def ab_track(frames, start, until, frame_s=.02):
    """20 ms frames with their times → one PCM track on the time line start..until (a missing frame is silence)."""
    count = int(round((until - start) / frame_s))
    track = bytearray(640 * count)
    for mono, frame in frames:
        i = int(round((mono - start) / frame_s))
        if 0 <= i < count and len(frame) == 640:
            track[640 * i:640 * (i + 1)] = frame
    return bytes(track)


def ab_multi_track(blocks, start):
    """The three microphones' blocks (epoch, first frame, [n × 3], phone time of the first frame) → one array by the
    board's frame numbers (the longest run of one epoch; a lost packet is zeros, marked in `valid`), and `t0` — its
    first sample on the A/B time line (s after the start), to find the same phrases as in the WAV tracks."""
    import numpy as np
    if not blocks:
        return None
    epoch = max({b[0] for b in blocks}, key=lambda e: sum(len(b[2]) for b in blocks if b[0] == e))
    mine = sorted((b for b in blocks if b[0] == epoch), key=lambda b: b[1])
    first = mine[0][1]
    n = mine[-1][1] + len(mine[-1][2]) - first
    if n <= 0 or n > 16000 * 600:
        return None
    x = np.zeros((n, 3), np.int16)
    valid = np.zeros(n, bool)
    for _, at, block, _ in mine:
        x[at - first:at - first + len(block)] = block
        valid[at - first:at - first + len(block)] = True
    t0 = float(np.median([mono - (at - first) / 16000. for _, at, _, mono in mine])) - start
    return {'x': x, 'valid': valid, 't0': np.float64(t0), 'first': np.int64(first)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    config = load(args.config)
    pin('voice', config)
    voice = Voice(config)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: voice.stopping.set())
    voice.run()


if __name__ == '__main__':
    main()
