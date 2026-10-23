"""The weather outside, from Open-Meteo (free, no key): «какая погода?», and a word about it in the morning.

Fetched in a worker thread at most every 30 min, never blocking a thought; the mind gets one Russian line.
"""
from __future__ import annotations

import json
import time
import urllib.request

URL = ('https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
       '&current=temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m'
       '&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code'
       '&timezone={tz}&forecast_days=2&wind_speed_unit=ms')
FRESH_S = 30 * 60

CODES = {0: 'ясно', 1: 'почти ясно', 2: 'переменная облачность', 3: 'пасмурно', 45: 'туман', 48: 'туман с изморозью',
         51: 'морось', 53: 'морось', 55: 'сильная морось', 56: 'ледяная морось', 57: 'ледяная морось',
         61: 'небольшой дождь', 63: 'дождь', 65: 'сильный дождь', 66: 'ледяной дождь', 67: 'ледяной дождь',
         71: 'небольшой снег', 73: 'снег', 75: 'сильный снег', 77: 'снежная крупа', 80: 'ливень', 81: 'ливни',
         82: 'сильные ливни', 85: 'снегопад', 86: 'сильный снегопад', 95: 'гроза', 96: 'гроза с градом',
         99: 'сильная гроза с градом'}


def describe(data):
    """One line for the mind from an Open-Meteo answer."""
    current, daily = data.get('current') or {}, data.get('daily') or {}
    parts = []
    if current:
        temp = current.get('temperature_2m')
        feels = current.get('apparent_temperature')
        sky = CODES.get(current.get('weather_code'), '')
        wind = current.get('wind_speed_10m')
        line = f'сейчас {temp:+.0f}°' if isinstance(temp, (int, float)) else 'сейчас'
        if isinstance(feels, (int, float)) and isinstance(temp, (int, float)) and abs(feels - temp) >= 3:
            line += f' (ощущается {feels:+.0f}°)'
        if sky:
            line += f', {sky}'
        if isinstance(wind, (int, float)):
            line += f', ветер {wind:.0f} м/с'
        parts.append(line)
    for index, word in ((0, 'сегодня'), (1, 'завтра')):
        try:
            low, high = daily['temperature_2m_min'][index], daily['temperature_2m_max'][index]
            rain = daily.get('precipitation_probability_max', [None, None])[index]
            sky = CODES.get(daily.get('weather_code', [None, None])[index], '')
        except (KeyError, IndexError, TypeError):
            continue
        line = f'{word} {low:+.0f}…{high:+.0f}°'
        if sky:
            line += f', {sky}'
        if isinstance(rain, (int, float)) and rain >= 30:
            line += f', осадки {rain:.0f}%'
        parts.append(line)
    return '; '.join(parts)


class Weather:
    def __init__(self, config):
        lat, lon = config.get('location', (55.75, 37.62))       # Moscow unless set
        tz = str(config.get('timezone', 'Europe/Moscow')).replace('/', '%2F')
        self.url = URL.format(lat=lat, lon=lon, tz=tz)
        self.text = ''
        self.current = {}                 # {'temp', 'code', 'words', 'high', 'low'} for the screen card
        self.at = 0.
        self.tried = 0.

    def due(self, now=None):
        now = time.time() if now is None else now
        return now - self.at > FRESH_S and now - self.tried > 300

    def fetch(self):
        """Blocking (run it in a worker): refresh the line; keeps the old one on failure."""
        self.tried = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url, headers={'User-Agent': 'bratan'}),
                                        timeout=8) as response:
                data = json.loads(response.read())
            text = describe(data)
        except Exception:
            return self.text
        if text:
            self.text, self.at = text, time.time()
            current, daily = data.get('current') or {}, data.get('daily') or {}
            try:
                self.current = {'temp': current.get('temperature_2m'), 'code': current.get('weather_code'),
                                'words': CODES.get(current.get('weather_code'), ''),
                                'high': daily['temperature_2m_max'][0], 'low': daily['temperature_2m_min'][0]}
            except (KeyError, IndexError, TypeError):
                self.current = {'temp': current.get('temperature_2m'), 'code': current.get('weather_code'),
                                'words': CODES.get(current.get('weather_code'), '')}
        return self.text

    def line(self, now=None):
        now = time.time() if now is None else now
        if not self.text or now - self.at > 6 * 3600:
            return ''
        return self.text
