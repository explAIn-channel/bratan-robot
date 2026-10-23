# face — Android-приложение

`ai.robotbrain.face` 5.14.6 (versionCode 97), minSdk 27, targetSdk 35, Java 8, AGP 8.6.1.
Работает на телефоне робота как домашний экран: рисует мордочку и даёт Python-процессам камеру, микрофон, динамик, экран, BLE-лампу и BLE-мост ушей. Наружу ничего не открывает: оба сервера слушают только `127.0.0.1`.

## Компоненты

| Класс | Что |
|---|---|
| `MainActivity` | запуск, лицо поверх экрана блокировки, яркость 0,16 |
| `BootReceiver` | старт после загрузки и обновления |
| `FaceServer` | HTTP `127.0.0.1:8766` |
| `MediaChannel` | TCP `127.0.0.1:8767`, поток звука |
| `CameraCapture`, `TripRecorder` | Camera2 и запись поездок |
| `AudioIO`, `LiveAudio` | микрофон, воспроизведение, потоковый PCM |
| `FaceView`, `ApprovedCorgiSpriteRig` | мордочка |
| `BleMoodLight`, `MoodLighting` | BLE-лампа |
| `BleEars` | мост ESP32 по BLE → UDP `127.0.0.1:5772` |
| `ProxyBroker` | локальный HTTP CONNECT-прокси для речевых сервисов |

## HTTP 8766

| Путь | Что |
|---|---|
| `/health`, `/state` | состояние |
| `/camera/frame.nv21`, `/camera/frame.jpg`, `/camera/rate`, `/camera/record`, `/camera/config` | кадр, частота 0,2–15 кадр/с, запись поездки |
| `/audio/cursor`, `/audio/stream.pcm`, `/audio/microphone.wav` | микрофон |
| `/audio/play.wav`, `/audio/play.mp3`, `/audio/playback`, `/audio/stop`, `/audio/live/*`, `/audio/effects` | динамик, шумодав и эхоподавление |
| `/emotion`, `/gaze`, `/display`, `/photo`, `/photo/clear` | лицо, взгляд, сон/пробуждение экрана, картинка на экране |
| `/mood-light/effect`, `/mood-light/preview` | лампа |
| `/proxy/*` | прокси и его настройка |

Доступ — `Authorization: Bearer <токен>` своей области (камера, микрофон, динамик, состояние, прокси); токены сравниваются по SHA-256. Запросы с заголовками `Origin` / `Sec-Fetch-*` отклоняются, у каждой области свой лимит частоты.

## TCP 8767

Клиент начинает с `AUTH <токен микрофона> <токен динамика>`. Кадры `[тип u8][длина u32 BE][данные]`:
клиент → приложение: 1 — PCM в динамик, 2 — стоп, 3 — `{"camera_fps": …}`; приложение → клиент: 10 — метка времени + 20 мс микрофона, 11 — JSON состояния.

## Камера и звук

- Камера: фронтальная (id 1), 640×480 YUV_420_888, без зеркала; поворот на 270° делает Python. JPEG (качество 84) кодируется только по запросу.
- Запись поездок: 1280×720, 30 кадр/с, H.264 4 Мбит/с + AAC 64 кбит/с, MP4.
- Микрофон: `VOICE_RECOGNITION`, 16 кГц моно, кольцевой буфер 12 с, системные NS и AEC.
- Динамик: AudioTrack + LoudnessEnhancer +10,5 дБ; WAV/MP3 через MediaPlayer, анимация рта по Visualizer. Синтеза речи в приложении нет.

## Мордочка

14 состояний: idle, listening, thinking, speaking, happy, laughing, curious, surprised, affectionate, sleepy, sad, concerned, annoyed, error.
Эмоции приходят из пяти источников со своим сроком жизни: голос (слушаю 25 с, думаю 60 с, говорю 120 с), реакция (4,5–14 с), событие (60 с), прямой приказ (15 с), тело. Они складываются в два слоя: занятие и настроение.
Моргание: закрытие 108–137 мс, пауза 12–27 мс, открытие 224–301 мс, 28 % морганий неполные. Засыпание 600 мс, пробуждение 250 мс. Восемь поз корги меняются в момент моргания.

## Лампа

Два протокола: Zengge/Telink mesh по GATT (ключ сессии AES) и ELK-BLEDOM. Цвет и яркость — по состоянию (`MoodLighting.java`), эффекты: rainbow, party, flash, warm, love, curious, calm, no, rgb. На базе — бирюзовая пульсация, со сном экрана лампа гаснет.

## Сборка

Gradle-проект без wrapper. Подпись release — переменные `ROBOTBRAIN_ANDROID_*`, токен ушей — `ROBOTBRAIN_EARS_TOKEN`, открытый ключ настройщика прокси — свойство Gradle. Учётные данные прокси хранятся только в Android Keystore телефона (AES-GCM), в коде их нет.
