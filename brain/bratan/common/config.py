"""One configuration for every process. Defaults here, overrides in config.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets'
HOME = Path(os.environ.get('BRATAN_HOME', '/opt/robotbrain/bratan'))
STATE = Path(os.environ.get('BRATAN_STATE', str(HOME / 'state')))
SECRETS = Path(os.environ.get('BRATAN_SECRETS', '/opt/robotbrain/secrets'))
MODELS = Path(os.environ.get('BRATAN_MODELS', '/opt/robotbrain/models'))

DEFAULTS = {
    # board link
    'board_host': '192.0.2.81', 'board_port': 8800, 'board_key': 'integrated-body-key',
    # maps and places
    'map_yaml': '/opt/robotbrain/maps/home/map.yaml',
    'rooms_json': '/opt/robotbrain/maps/home/rooms.json',
    'rooms_labels': '/opt/robotbrain/maps/home/rooms-labels.npy',
    # doorways the owner taught (07.10): the line to drive straight through, see body/passages.json
    'passages_json': str(Path(__file__).resolve().parent.parent / 'body' / 'passages.json'),
    'dock_pose': [0.0, 0.0, 0.0],
    # motion
    'max_speed': .24, 'cruise_speed': .24, 'turn_speed': .42,
    'lidar_pwm': 60, 'front_ir_mask': 6, 'front_ir_active_low': True,
    'undock_distance': .22, 'dock_staging_distance': .40, 'dock_overtravel': .05,
    # energy
    'board_reserve_voltage': 15.0, 'departure_voltage': 16.3,
    'phone_charge_limit_percent': 80, 'phone_charge_resume_percent': 75,
    # life
    'quiet_start_hour': 23, 'quiet_end_hour': 10, 'timezone': 'Europe/Moscow',
    'autonomy': False, 'name': 'Братан', 'pets': {'dog': 'Таффа'},
    'think_aloud': True,      # off the dock the first sentence of a key thought is said aloud (owner, 10.10)
    'people': ['Сергей', 'Марина'],
    # cores (big: 4-7, little: 0-3)
    'cpus': {'body': [4, 5], 'eyes': [5, 6, 7], 'voice': [4, 5], 'mind': [0, 1, 2, 3]},
    # models
    'llm_base_url': 'https://openrouter.ai/api/v1', 'llm_key': 'openrouter-api-key',
    # 18.10 (C52 bench on the robot's own prompts, work/checks/mind_bench.py): DeepSeek's 17.10 snapshot of V4.1 Flash
    # thinks as well as the old one and starts sooner (first word 0.74 s instead of 1.37 s); DeepInfra was the fastest
    # and the cheapest of its providers (Relace 0.77 s, Fireworks 0.9 s; StreamLake and Novita do not serve it).
    # Relace last: it also serves the V4 Pro fallback, at $8 per million words out and 8.5 s to the first one.
    'llm_providers': ['DeepInfra', 'Fireworks', 'Relace'],
    'llm_fast': 'deepseek/deepseek-v4.1-flash-1017',
    'llm_smart': 'deepseek/deepseek-v4-pro-0813',        # the fallback: V4 Pro's GA build (the April one was a preview)
    # The night's dream: the same Flash thinking first (medium) — its lessons were the most concrete and active of
    # V4 Pro (both builds), GLM-5.3 and Kimi K3 (V4 Pro twice wrote passive ones: «не ездить за ним», «не дёргаться»).
    'llm_dream': 'deepseek/deepseek-v4.1-flash-1017', 'llm_dream_reasoning': 'medium',
    # Talking (18.10, owner: «модели ещё должны быть максимально дешёвые… может, тот же DeepSeek для ответа голосом»;
    # work/checks/talk_bench.py, the real talk prompt, six phrases, OpenRouter's own price of each call): V4.1 Flash
    # 1017 — $0.00007 an answer (DeepInfra keeps 91 % of the prompt cached), first word 0.81 s (max 2 s), 12 of 12 in
    # format, lively Russian; Haiku 5.5 — $0.00051, 1.1 s. Cheaper ones lost: V4 Flash 0731 (a wrong fact, «камеры
    # свежего кадра нет»), Qwen 3.7 Flash (quotes around the words, made-up tool args), GLM 5.3 Flash (6.8 s).
    # Haiku stays as the fallback: another company's model if DeepSeek's providers all fail.
    'llm_talk': 'deepseek/deepseek-v4.1-flash-1017', 'llm_talk_fallback': 'anthropic/claude-haiku-5.5',
    # DeepSeek's own API with the owner's key (18.10: «апи именно дипсика, юзай оттуда»): the same models there by
    # their own names; OpenRouter only if DeepSeek does not answer. No key file — OpenRouter alone, as before.
    'llm_direct': {'base_url': 'https://api.deepseek.com', 'key': 'deepseek-api-key',
                   'models': {'deepseek/deepseek-v4.1-flash-1017': 'deepseek-flash',
                              'deepseek/deepseek-v4-pro-0813': 'deepseek-v4-pro'}},
    # V4.1 Flash sees images itself; the experimental vision model is being retired (DeepSeek routes it to V4.1) and
    # on the robot's album it took a corridor door for «a door to the street».
    'llm_vision': 'deepseek/deepseek-v4.1-flash-1017',
    'azure_speech_key': 'azure-speech-key', 'azure_speech_region': 'azure-speech-region',
    'live_key': 'openai-live-key', 'live_voice': 'coral', 'tts_model': 'gpt-4o-mini-tts',
    'speaker_volume': .7,     # everything to the speaker × this (18.10, the owner: «голос процентов на 30 тише»)
    # Ears in the body (17.10): three INMP441 on an ESP32 (work/ushi, voice/array.py) — where a sound came from and,
    # after the A/B check with the owner, his words. asr: 'phone' — only the phone's microphone hears words (as before);
    # 'auto' — the array while its stream is sound and we are not speaking, the phone while we speak (it cancels our
    # echo); 'array' — the array, the phone only when the stream fails.
    'ears': {'enabled': True, 'host': '192.0.2.85', 'asr': 'phone',
             'calibration': str(STATE / 'ears-calibration.json')},
}


def load(path=None) -> dict:
    path = Path(path or os.environ.get('BRATAN_CONFIG', str(HOME / 'config.json')))
    config = json.loads(json.dumps(DEFAULTS))
    if path.exists():
        override = json.loads(path.read_text(encoding='utf-8'))
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def secret(name: str) -> str:
    return (SECRETS / name).read_text().strip()


def pin(role: str, config: dict):
    """Pin this process to its cores; harmless where unsupported."""
    cpus = config.get('cpus', {}).get(role)
    if cpus and hasattr(os, 'sched_setaffinity'):
        # Android may isolate a core at any time; take the requested ones that are allowed now.
        for choice in (set(cpus), set(cpus[1:]), {4, 5, 6, 7}):
            try:
                os.sched_setaffinity(0, choice)
                return
            except OSError:
                continue


def write_json(path: Path, value) -> None:
    """Atomic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default
