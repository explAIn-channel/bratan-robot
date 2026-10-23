"""On the phone: Microsoft Edge TTS (ru-RU-DmitryNeural) — time to the first decoded audio, streamed MP3 decode."""
import asyncio, time
import av
import edge_tts


async def one(text):
    started = time.monotonic()
    codec = av.CodecContext.create('mp3', 'r')
    first_bytes = first_pcm = None
    samples = 0
    async for chunk in edge_tts.Communicate(text, 'ru-RU-DmitryNeural', rate='+8%').stream():
        if chunk['type'] != 'audio':
            continue
        if first_bytes is None:
            first_bytes = time.monotonic() - started
        for packet in codec.parse(chunk['data']):
            for frame in codec.decode(packet):
                if first_pcm is None:
                    first_pcm = time.monotonic() - started
                samples += frame.samples
    return first_bytes, first_pcm, time.monotonic() - started, samples / 24000


async def main():
    for text in ['Да?', 'Слышу тебя, я на базе, заряжаюсь.', 'Да вроде нормально. Стою на кухне, оглядываюсь. Скучаю по Таффе.'] * 2:
        b, p, total, audio = await one(text)
        print(f'first_bytes {b:.3f} first_pcm {p:.3f} total {total:.3f} audio {audio:.2f}s  {text[:30]}', flush=True)

asyncio.run(main())
