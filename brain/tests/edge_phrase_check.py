"""On the phone: why do some phrases get no audio? Library request vs ours (xml:lang), fresh connections."""
import asyncio, ssl, time, uuid
from xml.sax.saxutils import escape
import aiohttp
import edge_tts
from edge_tts.constants import WSS_URL, WSS_HEADERS, SEC_MS_GEC_VERSION
from edge_tts.drm import DRM
from edge_tts.communicate import date_to_string

CONFIG = ('X-Timestamp:{ts}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n'
          '{{"context":{{"synthesis":{{"audio":{{"metadataoptions":{{"sentenceBoundaryEnabled":"false",'
          '"wordBoundaryEnabled":"false"}},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}}}}}\r\n')
PHRASES = ['Не понял, повтори, пожалуйста.', 'Таффу сегодня видел в коридоре, он лежал у двери.',
           'Да вроде нормально, никуда не спешу.', 'Слышу тебя.']


async def library(text):
    t = time.monotonic()
    async for c in edge_tts.Communicate(text, 'ru-RU-DmitryNeural', rate='+8%').stream():
        if c['type'] == 'audio':
            return time.monotonic() - t
    return None


async def ours(text, lang, timeout=6):
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(f'{WSS_URL}&ConnectionId={uuid.uuid4().hex}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}'
                                      f'&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}', compress=15,
                                      headers=DRM.headers_with_muid(WSS_HEADERS), ssl=ssl.create_default_context())
        await ws.send_str(CONFIG.format(ts=date_to_string()))
        t = time.monotonic()
        await ws.send_str(f'X-RequestId:{uuid.uuid4().hex}\r\nContent-Type:application/ssml+xml\r\n'
                          f'X-Timestamp:{date_to_string()}Z\r\nPath:ssml\r\n\r\n'
                          f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{lang}'>"
                          f"<voice name='ru-RU-DmitryNeural'><prosody pitch='+0Hz' rate='+8%' volume='+0%'>"
                          f'{escape(text)}</prosody></voice></speak>')
        texts = []
        try:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    n = int.from_bytes(msg.data[:2], 'big')
                    if len(msg.data) > n + 2:
                        return time.monotonic() - t, texts
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    texts.append(msg.data.split('\r\n')[-1][:80] if 'Path:' in msg.data else msg.data[:80])
                    texts[-1] = [l for l in msg.data.split('\r\n') if l.startswith('Path:')] + [msg.data[-80:]]
                    if 'turn.end' in msg.data:
                        return None, texts
                else:
                    return None, texts + [str(msg.type)]
        except asyncio.TimeoutError:
            return None, texts + ['timeout']


async def main():
    for text in PHRASES:
        print('library', await library(text), text, flush=True)
        for lang in ('en-US', 'ru-RU'):
            print('  ours', lang, await ours(text, lang), flush=True)

asyncio.run(main())
