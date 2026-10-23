# brain — мозг на телефоне

Python-пакет `bratan`, работает на телефоне (Ubuntu в chroot) как один supervisor и три дочерних процесса, связанных шиной.

```
python -m bratan.main [--no-body] [--no-eyes] [--no-voice]
```

## Пакет

| Модуль | Файлы | Что |
|---|---|---|
| `common` | `bus.py`, `config.py`, `geometry.py`, `storage.py` | шина pub/sub на Unix-сокете, настройки по умолчанию + `config.json`, геометрия корпуса, контроль места на диске |
| `body` | `link.py`, `localize.py`, `grid.py`, `planner.py`, `maneuver.py`, `skills.py`, `guard.py`, `doors.py`, `scan.py`, `motion.py`, `feel.py`, `policy.py`, `passages.json` | связь с платой, локализация, карта, путь, навыки, защита, двери, «чувствую руки», нейросеть езды |
| `eyes` | `camera.py`, `detect.py`, `gesture.py`, `main.py` | кадры из приложения, YOLO, лица, трекинг, жесты, эмоции, видео поездок, альбом |
| `voice` | `wake.py`, `vad.py`, `hearing.py`, `azure.py`, `edge.py`, `local.py`, `overme.py`, `nullvoice.py`, `array.py`, `beam.py`, `denoise.py`, `sounds.py`, `audio.py`, `sfx.py`, `live.py` | имя, VAD, распознавание, синтез, перебивание, ноль на свой голос, решётка, звуки, воспроизведение |
| `mind` | `life.py`, `brain.py`, `llm.py`, `drives.py`, `affect.py`, `memory.py`, `world.py`, `hearing.py`, `games.py`, `screen.py`, `weather.py` | цикл жизни, мышление на LLM, потребности, настроение, память, модель мира, карта звуков, игры, экран, погода |
| `face` | `client.py` | клиент Android-приложения (HTTP 8766, токены по областям) |
| — | `cli.py` | `tail`, `pub`, `get`, `autonomy on|off` с ПК или телефона |

Подробный разбор алгоритмов — в [главном README](../README.ru.md#как-это-работает); история решений и источники — [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Настройки

`bratan/common/config.py` содержит значения по умолчанию; на роботе их перекрывает `/opt/robotbrain/state/config.json` (глубокое слияние).
Самое важное:

| Ключ | По умолчанию | Что |
|---|---|---|
| `board_host`, `board_port`, `board_key` | — , 8800, `integrated-body-key` | адрес платы и имя файла с её ключом в каталоге секретов |
| `map_yaml`, `rooms_json`, `rooms_labels` | `/opt/robotbrain/maps/home/…` | карта, комнаты, растр меток |
| `dock_pose` | `[0, 0, 0]` | поза базы на карте |
| `max_speed`, `turn_speed` | 0,24 м/с, 0,42 рад/с | скорости |
| `quiet_start_hour`, `quiet_end_hour`, `timezone` | 23, 10, `Europe/Moscow` | тихие часы |
| `people`, `pets` | `['Сергей', 'Марина']`, `{'dog': 'Таффа'}` | кого робот знает по имени |
| `ears` | `{'enabled': True, 'host': …, 'asr': 'phone'}` | решётка ESP32 |
| `speaker_volume` | 0,7 | громкость |
| `autonomy` | `False` | свободные поездки |

Ключи сервисов — файлы в `/opt/robotbrain/secrets` или `$BRATAN_SECRETS`.

## Деплой

| Скрипт | Что |
|---|---|
| `deploy/push.sh` | упаковать и залить релиз на телефон по ssh (хост `phone`), запустить `install.sh` |
| `deploy/install.sh` | распаковать в `releases/<время>`, переключить симлинк `app`, оставить 5 релизов |
| `deploy/safe_deploy.sh` | то же, но только когда робот стоит и не разговаривает |
| `deploy/switch.sh` | выбрать рабочую систему: `bratan` или `off` |
| `deploy/run.sh` | запуск из загрузочного хука Android |

## Инструменты

| Скрипт | Что |
|---|---|
| `tools/watch.py` | живая сводка состояния с ПК |
| `tools/liveness.py` | насколько робот «живой» по архиву событий |
| `tools/pull_videos.py`, `tools/video_agent.py` | забрать видео поездок на ПК |
| `tools/soul_backup.py` | резервная копия памяти, характера и альбома |
