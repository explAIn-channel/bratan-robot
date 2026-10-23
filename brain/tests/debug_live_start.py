"""On the phone: open a GPT-Live session the way the voice does and print the server's answer to session.start."""
import json, sys, time
sys.path.insert(0, '/opt/robotbrain/bratan/app')
from bratan.face import client as face
from bratan.voice.live import LiveSession
from bratan.common.persona import CHARACTER, LIVE_RULES


class Sink:
    speaking = False
    def play(self, pcm): pass
    def stop_playing(self): pass


key = open('/opt/robotbrain/secrets/openai-live-key').read().strip()
context = ''
if len(sys.argv) > 1:
    context = open(sys.argv[1], encoding='utf-8').read()
instructions = CHARACTER + '\n' + LIVE_RULES + ('\n\nЧТО ТЫ ЗНАЕШЬ О СЕБЕ СЕЙЧАС:\n' + context[:5000] if context else '')
print('instructions chars', len(instructions))
s = LiveSession(key, Sink(), instructions=lambda: instructions, voice='coral',
                on_event=lambda k, **d: print('event', k, d), proxy_url=face.proxy_url)
s.connect()
deadline = time.monotonic() + 5
while s.state == 'connecting' and time.monotonic() < deadline:
    time.sleep(.01)
print('state', s.state)
start = {'type': 'session.start', 'session': {
    'model': 'gpt-live-1', 'instructions': instructions,
    'audio': {'format': {'type': 'audio/pcm', 'rate': 16000}, 'output': {'voice': 'coral'}},
    'delegation': {'type': 'client'}, 'store': False}}
s._send(start)
for _ in range(5):
    print(s.ws.recv(timeout=6)[:600])
