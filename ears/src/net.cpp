// Сеть «ушей» (14.10, владелец: «продумай всё так, чтобы вайфай точно не падал»; переписано 17.10).
//
// 1. Wi-Fi без экономии энергии (WiFi.setSleep(false)): экономия — главная причина пропусков и «засыпаний» у ESP32.
// 2. Сторож связи раз в секунду: нет сети 15 с — reconnect; 60 с — заново с нуля; 10 мин без сети — перезагрузка.
//    Сеть ни разу не поднималась с включения (стенд без Wi-Fi, 15.10) — без переподключений и перезагрузок: это рвало
//    приём микрофонов; через 10 мин точка «Bratan-Ushi» для новой настройки (сменили пароль роутера).
// 3. Прошивка по воздуху с откатом: загрузчик держит прошлую прошивку. Новая считается рабочей, только когда 60 с
//    подряд держит Wi-Fi; не поднялась за 3 мин, зависла или перезагрузилась раньше — загрузчик возвращает прошлую.
//    Сломать «уши» прошивкой по воздуху так нельзя.
// 4. Сторож зависаний цикла (5 с) — перезагрузка вместо «тихо висит». На время приёма прошивки выключается.
// 5. UDP 5771, одна строка на запрос, ответ туда, откуда пришёл:
//      ping   → JSON: версия, время работы, сеть, сигнал, переподключения, прошивка на проверке, счётчики звука;
//      sub    → сырые 4 слота (пакеты BRE1 по 128 кадров) на этот адрес 3 с, повторять раз в секунду;
//      subv [слот|a] → голос одним каналом (пакеты BRV1 по 512 кадров = 32 мс, 16 бит); слот 0–3 или «a» — плата
//               выбирает живой и самый громкий (по умолчанию);
//      suba [слот|a] → то же, сжатое IMA-ADPCM (4 бита на отсчёт: 8 КБ/с против 128 КБ/с сырого потока);
//      subm [слоты] → несколько слотов в одном пакете BRM1 сжатием ADPCM (по умолчанию «023» — М1, М2, М3 владельца:
//               ~25 КБ/с, 31 пакет в секунду) — для направления на звук на телефоне;
//      unsub  → перестать слать этому адресу. Подписчиков до четырёх (панель, телефон, проверка — не мешают друг другу);
//      health → «H {JSON}»: что в каждом из 4 слотов (main.cpp, health_json);
//      pins   → «D …»: частоты такта и фронты на линиях данных (main.cpp, pin_report);
//      raw    → «X …»: 8 кадров сырых 32-битных слов всех 4 слотов (main.cpp, raw_dump);
//      restart → перезапустить оба I2S (для проверки сдвига между линиями после перезапуска).
//    Пакет BRE1: «BRE1», номер u32, первый кадр u32, кадров u16, каналов u16 (4), кадры × 4 int16 LE.
//    Пакеты BRV1 (голос) и BRM1 (несколько слотов), 20 байт заголовка: метка, номер u32, первый кадр u32,
//      кадров u16 (512), кодек u8 (0 — PCM16, 1 — IMA-ADPCM), слот u8 (BRV1) или маска слотов u8 (BRM1), эпоха u16,
//      каналов u8, 0 u8. Дальше по каналу (BRM1 — по возрастанию слота): PCM — кадры int16 LE; ADPCM — предсказание
//      int16, индекс шага u8, 0 u8, потом по 2 отсчёта в байте (младшая тетрада первая). Каждый пакет ADPCM
//      декодируется сам по себе. Все слоты одного пакета — из одних и тех же кадров.
// 6. Синий светодиод платы (GPIO2): горит — Wi-Fi есть; мигает раз в секунду — ищет сеть; часто — открыта точка
//    Bratan-Ushi или ждёт кнопку WPS.
#include "net.h"
#include "ble.h"

#include <ArduinoOTA.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <WiFiUdp.h>
#include <esp_ota_ops.h>
#include <esp_wps.h>

#ifndef OTA_PASS
#define OTA_PASS ""
#endif

static const uint16_t UDP_PORT = 5771;
static const int PIN_LED = 2;
// Точка «Bratan-Ushi» — 10 минут (16.10: владелец настраивает Wi-Fi с телефона, 3 минут мало).
static const uint32_t RECONNECT_MS = 15000, FRESH_MS = 60000, PORTAL_MS = 600000, REBOOT_MS = 600000;
static const uint32_t TRIAL_OK_MS = 60000, TRIAL_FAIL_MS = 180000, SUB_MS = 3000;

static WiFiManager manager;
static WiFiUDP udp;
static bool ota_ready = false, ever_online = false, portal_open = false, trial = false;
static uint32_t offline_since = 0, online_since = 0, last_check = 0, last_kick = 0, booted = 0;
static uint32_t drops = 0, reconnects = 0, sent_packets = 0, send_errors = 0, send_retries = 0;
static uint8_t last_reason = 0;
static void (*ota_start_hook)() = nullptr;
static void (*ota_fail_hook)() = nullptr;

enum SubKind : uint8_t { SUB_RAW, SUB_PCM, SUB_ADPCM, SUB_MULTI };
static const int RAW_FRAMES = 128, VOICE_FRAMES = 512;
struct Subscriber {
  IPAddress ip;
  uint16_t port = 0;
  uint32_t at = 0;
  uint8_t kind = SUB_RAW;
  int8_t slot = -1;                                        // голос: слот 0–3, −1 — плата выбирает сама
  uint8_t mask = 0x0D;                                     // несколько слотов (subm): по умолчанию 0, 2, 3
  uint32_t seq = 0;
  int16_t pred[4] = {0, 0, 0, 0};                          // состояние кодера ADPCM по слотам
  uint8_t step[4] = {0, 0, 0, 0};
};
static Subscriber subs[4];
static volatile bool streaming = false;
// Общее окно звука на 512 кадров (32 мс) для голоса и нескольких слотов: копится из двух блоков приёма.
static int16_t window[VOICE_FRAMES][4];
static int window_fill = 0;
static uint32_t window_first = 0, window_epoch = 0;

// Сказать Arduino, что прошивку «подтвердим» сами, когда сеть докажет, что работает.
extern "C" bool verifyRollbackLater() { return true; }

// WPS по кнопке роутера (15.10): сеть подключается без ввода пароля — роутер сам отдаёт его плате.
static bool wps_running = false;

void net_start_wps() {
  if (wps_running) return;
  if (portal_open) { manager.stopConfigPortal(); portal_open = false; }
  WiFi.disconnect(false, false);
  WiFi.mode(WIFI_STA);
  esp_wps_config_t config = WPS_CONFIG_INIT_DEFAULT(WPS_TYPE_PBC);
  esp_wifi_wps_enable(&config);
  esp_wifi_wps_start(0);
  wps_running = true;
  Serial.println("I WPS: жду нажатия кнопки WPS на роутере (2 минуты)");
}

static void on_event(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WPS_ER_SUCCESS) {
    esp_wifi_wps_disable();
    wps_running = false;
    Serial.println("I WPS: роутер пустил, подключаюсь");
    WiFi.persistent(true);
    WiFi.begin();                                         // пароль от роутера сохранён в самой плате
    return;
  }
  if (event == ARDUINO_EVENT_WPS_ER_FAILED || event == ARDUINO_EVENT_WPS_ER_TIMEOUT ||
      event == ARDUINO_EVENT_WPS_ER_PIN) {
    esp_wifi_wps_disable();
    wps_running = false;
    Serial.println("E WPS не вышел (кнопку не нажали за 2 минуты или роутер не поддерживает)");
    return;
  }
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    if (online_since) drops++;
    online_since = 0;
    last_reason = info.wifi_sta_disconnected.reason;
    if (!offline_since) offline_since = millis();
  } else if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    online_since = millis();
    offline_since = 0;
    ever_online = true;
  }
}

static void start_ota() {
  if (ota_ready || strlen(OTA_PASS) < 8) return;          // без пароля прошивка по воздуху выключена
  ArduinoOTA.setHostname("bratan-ushi");
  ArduinoOTA.setPassword(OTA_PASS);
  ArduinoOTA.setRebootOnSuccess(true);
  ArduinoOTA.onStart([]() {
    disableLoopWDT();                                      // приём идёт дольше 5 с
    if (ota_start_hook) ota_start_hook();
    Serial.println("I приём прошивки по Wi-Fi");
  });
  ArduinoOTA.onError([](ota_error_t e) {
    Serial.printf("E прошивка по Wi-Fi не удалась (%u), работаю на прежней\n", e);
    enableLoopWDT();
    if (ota_fail_hook) ota_fail_hook();
  });
  ArduinoOTA.begin();
  ota_ready = true;
}

void net_setup(void (*on_ota_start)(), void (*on_ota_fail)()) {
  ota_start_hook = on_ota_start;
  ota_fail_hook = on_ota_fail;
  booted = millis();
  pinMode(PIN_LED, OUTPUT);
  esp_ota_img_states_t state;
  const esp_partition_t *running = esp_ota_get_running_partition();
  trial = esp_ota_get_state_partition(running, &state) == ESP_OK && state == ESP_OTA_IMG_PENDING_VERIFY;
  WiFi.onEvent(on_event);
  WiFi.mode(WIFI_STA);
  WiFi.setHostname("bratan-ushi");
  WiFi.setSleep(false);                                    // без экономии энергии: связь ровнее
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  manager.setConfigPortalBlocking(false);
  manager.setConfigPortalTimeout(PORTAL_MS / 1000);
  manager.setConnectTimeout(20);
  manager.setDebugOutput(false);
  // Сеть уже знакома — просто подключиться; незнакома — точка «Bratan-Ushi» на 10 минут.
  portal_open = !manager.autoConnect("Bratan-Ushi");
  udp.begin(UDP_PORT);
  enableLoopWDT();
  if (trial) Serial.println("I новая прошивка на проверке: подтвердится после 60 с устойчивой сети");
}

bool net_online() { return WiFi.status() == WL_CONNECTED; }

static int active_subscribers() {
  int n = 0;
  for (const auto &s : subs) n += s.port && millis() - s.at < SUB_MS;
  return n;
}

String net_status() {
  esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
  esp_ota_get_state_partition(esp_ota_get_running_partition(), &state);
  char line[1024];
  snprintf(line, sizeof(line),
           "{\"fw\":\"%s\",\"up_s\":%lu,\"online\":%s,\"ip\":\"%s\",\"rssi\":%d,\"drops\":%lu,\"reconnects\":%lu,"
           "\"last_reason\":%u,\"trial\":%s,\"heap\":%u,\"streaming\":%s,\"subs\":%d,\"sent\":%lu,\"send_errors\":%lu,"
           "\"send_retries\":%lu,\"reset\":%d,\"heap_min\":%u,%s,%s}",
           FW_VERSION, (unsigned long)(millis() / 1000), net_online() ? "true" : "false",
           WiFi.localIP().toString().c_str(), net_online() ? WiFi.RSSI() : 0, (unsigned long)drops,
           (unsigned long)reconnects, last_reason, trial ? "true" : "false", ESP.getFreeHeap(),
           streaming ? "true" : "false", active_subscribers(), (unsigned long)sent_packets,
           (unsigned long)send_errors, (unsigned long)send_retries, (int)esp_reset_reason(), ESP.getMinFreeHeap(),
           audio_counters().c_str(), ble_status().c_str());
  return String(line);
}

void net_forget_wifi() {
  manager.resetSettings();
  Serial.println("I Wi-Fi забыт: перезагрузка, точка Bratan-Ushi");
  delay(200);
  ESP.restart();
}

static void watchdog() {
  uint32_t now = millis();
  if (wps_running) return;                                 // идёт WPS: не мешать переподключениями
  if (net_online()) {
    if (!online_since) online_since = now;
    offline_since = 0;
    start_ota();
    if (trial && now - online_since >= TRIAL_OK_MS) {
      esp_ota_mark_app_valid_cancel_rollback();             // новая прошивка доказала, что держит сеть
      trial = false;
      Serial.println("I новая прошивка подтверждена");
    }
    return;
  }
  if (!offline_since) offline_since = now;
  uint32_t off = now - offline_since;
  if (trial && now - booted >= TRIAL_FAIL_MS) {
    Serial.println("E новая прошивка не подняла Wi-Fi за 3 мин — откат на прежнюю");
    delay(100);
    esp_ota_mark_app_invalid_rollback_and_reboot();
  }
  if (!ever_online && !trial) {
    // Сеть ещё ни разу не поднималась: не дёргать радио и не перезагружаться. Точка «Bratan-Ushi» — по таймеру.
    // (Новая прошивка на проверке, наоборот, переподключается изо всех сил: у неё 3 минуты.)
    if (!portal_open && now - booted >= PORTAL_MS && manager.getWiFiIsSaved()) {
      manager.startConfigPortal("Bratan-Ushi");
      portal_open = true;
    }
    return;
  }
  if (off >= REBOOT_MS) {
    Serial.println("E 10 мин без Wi-Fi — перезагрузка");
    delay(100);
    ESP.restart();
  }
  if (off >= FRESH_MS && now - last_kick >= FRESH_MS) {
    last_kick = now;
    reconnects++;
    WiFi.disconnect(false, false);
    WiFi.begin();                                          // с нуля, с сохранёнными сетью и паролем
  } else if (off >= RECONNECT_MS && now - last_kick >= RECONNECT_MS) {
    last_kick = now;
    reconnects++;
    WiFi.reconnect();
  }
}

// «sub», «subv 2», «suba a», «subm 023»: вид потока, слот голоса (−1 — плата выбирает сама), слоты многоканального.
static void subscribe(const IPAddress &ip, uint16_t port, const char *request) {
  uint8_t kind = strncmp(request, "subv", 4) == 0   ? SUB_PCM
                 : strncmp(request, "suba", 4) == 0 ? SUB_ADPCM
                 : strncmp(request, "subm", 4) == 0 ? SUB_MULTI
                                                    : SUB_RAW;
  int8_t want = -1;
  uint8_t mask = 0;
  if (kind != SUB_RAW) {
    const char *arg = request + 4;
    while (*arg == ' ') arg++;
    if (kind == SUB_MULTI)
      for (; *arg >= '0' && *arg <= '3'; arg++) mask |= 1 << (*arg - '0');
    else if (*arg >= '0' && *arg <= '3')
      want = *arg - '0';
  }
  if (kind == SUB_MULTI && !mask) mask = 0x0D;
  uint32_t now = millis();
  Subscriber *slot = nullptr;
  for (auto &s : subs)
    if (s.port == port && s.ip == ip) slot = &s;
  bool fresh = slot == nullptr;
  for (auto &s : subs)
    if (!slot && (!s.port || now - s.at >= SUB_MS)) slot = &s;
  if (!slot) {                                             // все места заняты живыми — сменить самого давнего
    slot = &subs[0];
    for (auto &s : subs)
      if (now - s.at > now - slot->at) slot = &s;
  }
  if (fresh || slot->kind != kind || slot->slot != want || (kind == SUB_MULTI && slot->mask != mask)) {
    slot->seq = 0;
    for (int ch = 0; ch < 4; ch++) { slot->pred[ch] = 0; slot->step[ch] = 0; }
  }
  slot->ip = ip;
  slot->port = port;
  slot->at = now;
  slot->kind = kind;
  slot->slot = want;
  if (kind == SUB_MULTI) slot->mask = mask;
}

static void unsubscribe(const IPAddress &ip, uint16_t port) {
  for (auto &s : subs)
    if (s.port == port && s.ip == ip) s.port = 0;
}

static void reply(const String &text) {
  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.write((const uint8_t *)text.c_str(), text.length());
  udp.endPacket();
}

static void answer_requests() {
  for (int k = 0; k < 8; k++) {                            // все накопившиеся запросы, но не больше 8 за проход
    if (udp.parsePacket() <= 0) return;
    char request[16] = {0};
    udp.read(request, sizeof(request) - 1);
    if (strncmp(request, "unsub", 5) == 0) unsubscribe(udp.remoteIP(), udp.remotePort());
    else if (strncmp(request, "sub", 3) == 0) subscribe(udp.remoteIP(), udp.remotePort(), request);
    else if (strncmp(request, "health", 6) == 0) reply("H " + health_json());
    else if (strncmp(request, "pins", 4) == 0) reply(pin_report());
    else if (strncmp(request, "raw", 3) == 0) reply(raw_dump());
    else if (strncmp(request, "restart", 7) == 0) { audio_restart(); reply("I перезапуск I2S"); }
    else if (strncmp(request, "ping", 4) == 0) reply(net_status());
  }
}

bool net_streaming() { return streaming; }

// 17.10: при слабом сигнале очередь передачи Wi-Fi забивается и отправка отказывает (на ushi-6 — 11 % пакетов).
// Пакет не выбрасывается сразу: до 3 повторов через 1 мс. Ждёт только цикл — приём звука идёт своей задачей.
static void send_to(const Subscriber &s, const uint8_t *data, size_t len) {
  bool ok = false;
  for (int attempt = 0; attempt < 4 && !ok; attempt++) {
    if (attempt) { send_retries++; vTaskDelay(1); }
    ok = udp.beginPacket(s.ip, s.port) && udp.write(data, len) == len && udp.endPacket();
  }
  if (ok) sent_packets++;
  else send_errors++;
}

// IMA-ADPCM: 4 бита на отсчёт, шаг подстраивается под громкость. Состояние кодера в начале каждого пакета.
static const int16_t ADPCM_STEP[89] = {
    7,     8,     9,     10,    11,    12,    13,    14,    16,    17,    19,    21,    23,    25,    28,
    31,    34,    37,    41,    45,    50,    55,    60,    66,    73,    80,    88,    97,    107,   118,
    130,   143,   157,   173,   190,   209,   230,   253,   279,   307,   337,   371,   408,   449,   494,
    544,   598,   658,   724,   796,   876,   963,   1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
    2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,  5894,  6484,  7132,  7845,  8630,
    9493,  10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767};
static const int8_t ADPCM_INDEX[16] = {-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8};

static uint8_t adpcm_code(int16_t sample, int16_t &pred, uint8_t &step_index) {
  int step = ADPCM_STEP[step_index];
  int diff = sample - pred;
  uint8_t code = 0;
  if (diff < 0) { code = 8; diff = -diff; }
  int delta = step >> 3;
  if (diff >= step) { code |= 4; diff -= step; delta += step; }
  step >>= 1;
  if (diff >= step) { code |= 2; diff -= step; delta += step; }
  step >>= 1;
  if (diff >= step) { code |= 1; delta += step; }
  int p = pred + ((code & 8) ? -delta : delta);
  pred = (int16_t)(p > 32767 ? 32767 : p < -32768 ? -32768 : p);
  int idx = step_index + ADPCM_INDEX[code];
  step_index = (uint8_t)(idx < 0 ? 0 : idx > 88 ? 88 : idx);
  return code;
}

// Заголовок пакета голоса/нескольких слотов (20 байт): метка, номер, первый кадр, кадров, кодек, слот или маска,
// эпоха, число каналов.
static size_t voice_header(uint8_t *packet, const char *tag, Subscriber &s, uint8_t codec, uint8_t slot_or_mask,
                           uint8_t channels) {
  memcpy(packet, tag, 4);
  uint16_t n = VOICE_FRAMES, epoch16 = (uint16_t)window_epoch;
  memcpy(packet + 4, &s.seq, 4);
  memcpy(packet + 8, &window_first, 4);
  memcpy(packet + 12, &n, 2);
  packet[14] = codec;
  packet[15] = slot_or_mask;
  memcpy(packet + 16, &epoch16, 2);
  packet[18] = channels;
  packet[19] = 0;
  return 20;
}

// Один слот окна сжатием ADPCM: состояние кодера (4 байта) и 256 байт тетрад.
static size_t adpcm_block(uint8_t *out, Subscriber &s, int slot) {
  memcpy(out, &s.pred[slot], 2);
  out[2] = s.step[slot];
  out[3] = 0;
  uint8_t *p = out + 4;
  for (int i = 0; i < VOICE_FRAMES; i += 2) {
    uint8_t lo = adpcm_code(window[i][slot], s.pred[slot], s.step[slot]);
    uint8_t hi = adpcm_code(window[i + 1][slot], s.pred[slot], s.step[slot]);
    *p++ = lo | (hi << 4);
  }
  return 4 + VOICE_FRAMES / 2;
}

static uint8_t packet_buf[20 + 4 * (4 + VOICE_FRAMES / 2) + VOICE_FRAMES * 2];

// Пакет окна для получателя (голос или несколько слотов) в packet_buf; возвращает длину.
static size_t build_window(Subscriber &s) {
  uint8_t *packet = packet_buf;
  size_t len;
  if (s.kind == SUB_MULTI) {
    uint8_t channels = 0;
    for (int ch = 0; ch < 4; ch++) channels += (s.mask >> ch) & 1;
    len = voice_header(packet, "BRM1", s, 1, s.mask, channels);
    for (int ch = 0; ch < 4; ch++)
      if ((s.mask >> ch) & 1) len += adpcm_block(packet + len, s, ch);
  } else {
    int slot = s.slot >= 0 ? s.slot : voice_slot_auto();
    len = voice_header(packet, "BRV1", s, s.kind == SUB_ADPCM ? 1 : 0, (uint8_t)slot, 1);
    if (s.kind == SUB_ADPCM) {
      len += adpcm_block(packet + len, s, slot);
    } else {
      for (int i = 0; i < VOICE_FRAMES; i++) memcpy(packet + len + 2 * i, &window[i][slot], 2);
      len += VOICE_FRAMES * 2;
    }
  }
  s.seq++;
  return len;
}

static void send_window(Subscriber &s) {
  size_t len = build_window(s);
  send_to(s, packet_buf, len);
}

// Телефон по Bluetooth (ble.cpp): получатель трёх микрофонов, как «subm 023», со своим счётом пакетов и кодером.
static Subscriber ble_sub;
static bool ble_was_streaming = false;

void net_audio(const AudioChunk &c) {
  if (!streaming) return;
  uint32_t now = millis();
  bool want_window = false;
  for (auto &s : subs) {
    if (!s.port || now - s.at >= SUB_MS) continue;
    if (s.kind != SUB_RAW) { want_window = true; continue; }
    static uint8_t packet[16 + RAW_FRAMES * 4 * 2];
    for (int part = 0; part < CHUNK / RAW_FRAMES; part++) {
      memcpy(packet, "BRE1", 4);
      uint32_t first = c.first + part * RAW_FRAMES;
      uint16_t n = RAW_FRAMES, chans = 4;
      memcpy(packet + 4, &s.seq, 4);
      memcpy(packet + 8, &first, 4);
      memcpy(packet + 12, &n, 2);
      memcpy(packet + 14, &chans, 2);
      memcpy(packet + 16, c.s + part * RAW_FRAMES * 4, RAW_FRAMES * 4 * 2);
      send_to(s, packet, sizeof(packet));
      s.seq++;
    }
  }
  bool ble = ble_streaming();
  if (ble && !ble_was_streaming) {                        // новый подписчик по Bluetooth: кодер с нуля
    ble_sub = Subscriber();
    ble_sub.kind = SUB_MULTI;
    ble_sub.mask = 0x0D;
  }
  ble_was_streaming = ble;
  if (!want_window && !ble) { window_fill = 0; return; }
  // Окно 512 кадров: новая эпоха или разрыв по кадрам — копим заново.
  if (window_fill && (c.epoch != window_epoch || c.first != window_first + window_fill)) window_fill = 0;
  if (!window_fill) { window_first = c.first; window_epoch = c.epoch; }
  memcpy(window[window_fill], c.s, CHUNK * 4 * sizeof(int16_t));
  window_fill += CHUNK;
  if (window_fill < VOICE_FRAMES) return;
  window_fill = 0;
  for (auto &s : subs)
    if (s.port && now - s.at < SUB_MS && s.kind != SUB_RAW) send_window(s);
  if (ble) ble_send(packet_buf, build_window(ble_sub));
}

// ushi-24 (19.10, владелец: «отключи на плате обе лампочки»): синяя всегда погашена — плата в корпусе, её свет не нужен.
// Красная — индикатор питания на 3,3 В, программно не гасится.
static void led_tick() { digitalWrite(PIN_LED, LOW); }

void net_tick() {
  feedLoopWDT();
  manager.process();
  if (millis() - last_check >= 1000) {
    last_check = millis();
    watchdog();
    // ushi-22 tried ESP_COEX_PREFER_BT while the sound went by Bluetooth alone: Bluetooth lost 0–3 % instead of 15–37 %,
    // but Wi-Fi got no air at all — no ping, no OTA, the board lost from the network (18.10 19:00). Never again: the
    // radio stays balanced (ushi-23); a board without Wi-Fi cannot be reflashed without its USB.
  }
  if (ota_ready) ArduinoOTA.handle();
  if (net_online()) answer_requests();
  streaming = (net_online() && active_subscribers() > 0) || ble_streaming();   // Bluetooth — и без Wi-Fi
  led_tick();
}
