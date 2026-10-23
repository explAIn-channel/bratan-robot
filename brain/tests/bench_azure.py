"""On the phone: Azure Speech TTS (official) — first audio byte over one kept-alive HTTPS connection."""
import http.client, ssl, time
from xml.sax.saxutils import escape

key = open('/opt/robotbrain/secrets/azure-speech-key').read().strip()
region = open('/opt/robotbrain/secrets/azure-speech-region').read().strip()
host = f'{region}.tts.speech.microsoft.com'
conn = http.client.HTTPSConnection(host, 443, timeout=10, context=ssl.create_default_context())
t = time.monotonic()
conn.connect()
print(f'connect {time.monotonic() - t:.3f}s')
phrases = ['Слышу тебя.', 'Не понял, повтори, пожалуйста.', 'Таффу сегодня видел в коридоре, он лежал у двери.',
           'Да вроде нормально, никуда не спешу.', 'Я на базе, заряжаюсь.']
for i in range(2):
    for text in phrases:
        ssml = ("<speak version='1.0' xml:lang='ru-RU'><voice name='ru-RU-DmitryNeural'>"
                f"<prosody rate='+8%'>{escape(text)}</prosody></voice></speak>").encode()
        t = time.monotonic()
        conn.request('POST', '/cognitiveservices/v1', ssml, {
            'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'raw-16khz-16bit-mono-pcm', 'User-Agent': 'bratan'})
        r = conn.getresponse()
        first = None
        size = 0
        while True:
            chunk = r.read1(4096) if hasattr(r, 'read1') else r.read(4096)
            if not chunk:
                break
            if first is None:
                first = time.monotonic() - t
            size += len(chunk)
        print(f'{r.status} first {first if first is None else round(first, 3)}s total {time.monotonic() - t:.3f}s '
              f'audio {size / 32000:.2f}s  {text[:30]}', flush=True)
    time.sleep(1)
