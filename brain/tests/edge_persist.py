"""On the phone: can one Edge TTS WebSocket serve many phrases, and how long does it stay open idle?

Prints connect time and, per request on the same socket, time to the first audio byte.
"""
import asyncio, ssl, sys, time, uuid
import aiohttp
from edge_tts.constants import WSS_URL, WSS_HEADERS, SEC_MS_GEC_VERSION
from edge_tts.drm import DRM
from edge_tts.communicate import date_to_string

VOICE = 'ru-RU-DmitryNeural'
CONFIG = ('X-Timestamp:{ts}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n'
          '{{"context":{{"synthesis":{{"audio":{{"metadataoptions":{{"sentenceBoundaryEnabled":"false",'
          '"wordBoundaryEnabled":"false"}},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}}}}}\r\n')


def ssml(text):
    return (f"X-RequestId:{uuid.uuid4().hex}\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:{date_to_string()}Z\r\n"
            "Path:ssml\r\n\r\n<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>"
            f"<voice name='{VOICE}'><prosody pitch='+0Hz' rate='+8%' volume='+0%'>{text}</prosody></voice></speak>")


async def request(ws, text):
    t = time.monotonic()
    await ws.send_str(ssml(text))
    first, size = None, 0
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            n = int.from_bytes(msg.data[:2], 'big')
            if len(msg.data) > n + 2:
                if first is None:
                    first = time.monotonic() - t
                size += len(msg.data) - n - 2
        elif msg.type == aiohttp.WSMsgType.TEXT:
            if 'Path:turn.end' in msg.data:
                return first, time.monotonic() - t, size
        else:
            raise RuntimeError(f'ws {msg.type}')
    raise RuntimeError('closed')


async def main():
    gaps = [float(g) for g in sys.argv[1:]] or [0, 0, 5, 30, 60, 180]
    async with aiohttp.ClientSession() as session:
        t = time.monotonic()
        ws = await session.ws_connect(f'{WSS_URL}&ConnectionId={uuid.uuid4().hex}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}'
                                      f'&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}', compress=15,
                                      headers=DRM.headers_with_muid(WSS_HEADERS), ssl=ssl.create_default_context(),
                                      heartbeat=None)
        print(f'connect {time.monotonic() - t:.3f}s', flush=True)
        await ws.send_str(CONFIG.format(ts=date_to_string()))
        for gap in gaps:
            await asyncio.sleep(gap)
            try:
                first, total, size = await request(ws, 'Слышу тебя, я на базе, заряжаюсь.')
                print(f'after idle {gap:5.0f}s: first {first:.3f}s total {total:.3f}s bytes {size}', flush=True)
            except Exception as error:
                print(f'after idle {gap:5.0f}s: FAILED {type(error).__name__} {error} closed={ws.closed} '
                      f'code={ws.close_code}', flush=True)
                return

asyncio.run(main())
