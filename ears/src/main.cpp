// «Уши Братана» (ushi-12, 17.10.2026): 4×INMP441 на двух I2S с общим тактом, звук по Wi-Fi, самопроверка каналов.
//
// Проводка (записка bratan-ushi-mikrofony.md):
//   GPIO26 = SCK (общий), GPIO25 = WS (общий), GPIO33 = SD1 (М1 левый + М2 правый), GPIO32 = SD2 (М3 левый + М4 правый)
// I2S0 — ведущий: выдаёт SCK/WS и читает SD1. I2S1 — ведомый: читает тот же SCK/WS с тех же ножек и SD2.
// Ножки 26/25 одновременно выход I2S0 и вход I2S1: стандартная настройка драйвера так не умеет, маршрут через матрицу
// GPIO прописан вручную (route_shared_clock). Ведомый запускается раньше ведущего: он ждёт первый такт.
//
// Аудит 17.10: раньше приём звука, Wi-Fi и USB шли в одном цикле. Любая задержка — отправка по слабому Wi-Fi, запись
// «c» по USB (1 с), проверка ножек (240 мс) — переполняла буфер I2S (~110 мс): звук терялся, а две линии могли
// разъехаться на целый блок (16 мс) до следующего перезапуска. Проверка ножек к тому же снимала подтяжку с линий данных,
// и пустые провода давали мусор во всех каналах. Теперь приём — отдельная задача (audio_task) выше по приоритету:
// Wi-Fi и USB её не задерживают; переполнение или обрыв такта ловится и оба I2S перезапускаются вместе (кадр с нуля,
// «эпоха» +1 — приёмник заново меряет сдвиг между линиями). Ножки меряет аппаратный счётчик (PCNT), подтяжки не трогает.
//
// ushi-12/13 (17.10, микрофоны в роботе):
// - Фильтр ниже 60 Гц на всём звуке: INMP441 отдаёт медленный дрейф (ниже 5 Гц). Шум и выше 100 Гц остался
//   (−48…−55 дБ полной шкалы, тона 95,5 / 167 / 860 Гц): в комнате или в корпусе что-то звучит/дрожит — ищется.
//   Уровни на 6 дБ выше настоящих: I2S берёт слово на такт позже (младший бит 24-битного слова всегда 0, бит 0
//   слота — старший бит соседнего), то есть отсчёт вдвое больше; форма звука верная.
// - Сдвиг между линиями: в записи из робота (ushi-11, после аварийных перезапусков) М1 (I2S0) отставал от М2/М3
//   (I2S1) на ~340 кадров (21 мс); после чистого перезапуска сдвига нет. Старт обеих линий теперь неразрывный
//   (start_aligned), приём — с приоритетом 22 (выше всех задач Wi-Fi на ядре 1). В самоотчёте: «первый звук» каждой
//   линии после перезапуска (wake) и паузы приёма (gap/stall).
// - Голос для распознавания (UDP subv/suba): один канал вместо четырёх, пакеты по 32 мс, можно сжатием IMA-ADPCM
//   (4 бита на отсчёт, 8 КБ/с); несколько слотов одним пакетом для направления (subm) — в корпусе сигнал Wi-Fi
//   −74…−81 дБм, полный поток 128 КБ/с там терял до 91 %, голос 32 КБ/с — меньше 1 %.
//
// Самопроверка каналов (команда h, UDP «health»): плата сама говорит, что в каждом слоте. INMP441 выдаёт 24 бита и
// отпускает линию на остальные 8 бит слота — подтяжка держит их нулями (бит 0 — старший бит соседа, см. empty_word).
//   none  — слова пустые: микрофона нет, нет питания, SD не туда или L/R не в тот слот;
//   float — мусор в битах 5..1, которые микрофон не выдаёт: линия висит без микрофона и ловит наводки (или чужой провод);
//   ghost — в слоте те же 24 бита, что в соседнем (тот же кадр или соседний): своего микрофона нет, сюда «перетекает»
//           сосед с болтающимся L/R;
//   spill — сам микрофон жив, но перетекает в соседний слот (его L/R болтается);
//   hop   — один микрофон скачет между левым и правым слотом линии (17.10: каждые 10 мс, в такт сети 50 Гц) — его
//           L/R ни к чему не подключён и ловит наводку;
//   gaps  — звук то есть, то нет (плохой контакт); noise — громкий белый шум без формы (битые данные);
//   ok    — живой микрофон.
//
// Команды по USB (921600 бод), одна буква:
//   l — уровни 5 раз в секунду (строка L, по умолчанию), s — тишина, h — здоровье каналов (строка H, JSON);
//   c — запись 0,75 с всех 4 каналов на ПК (CAP <кадров> 4 <частота>, int16 LE, END); r — перезапуск обоих I2S;
//   d — такты и линии данных (строка D); x — сырые 32-битные слова (строка X); i — прошивка и сеть;
//   w — забыть Wi-Fi; p — WPS.
// Сеть — net.cpp: Wi-Fi со сторожем, прошивка по воздуху с откатом, UDP 5771 (ping, sub, subv, suba, health, pins,
// raw, restart).
#include <Arduino.h>
#include "net.h"
#include "ble.h"
#include <algorithm>
#include <driver/i2s.h>
#include <driver/gpio.h>
#include <driver/pcnt.h>
#include <esp_timer.h>
#include <soc/i2s_periph.h>
#include <rom/gpio.h>

static const int PIN_SCK = 26, PIN_WS = 25, PIN_SD1 = 33, PIN_SD2 = 32;
static const int RATE = 16000;
static const int CAPTURE = 12000;                   // 0,75 с
static const int FAST_CHUNKS = 12, SLOW_CHUNKS = 31; // окна: уровни ~0,2 с, здоровье ~0,5 с
static int32_t buf0[CHUNK * 2], buf1[CHUNK * 2];
static float hp[CHUNK][4];                          // звук после фильтра ниже 60 Гц (шкала 24 бит)
static QueueHandle_t events[2];

// Общее для задачи приёма и цикла (обе на ядре 1).
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static volatile bool paused = false, restart_req = false;
static volatile uint32_t frames_total = 0, epoch = 0, overflows = 0, i2s_failures = 0, queue_drops = 0;
static volatile uint32_t gap_max_us = 0, stalls = 0;      // паузы между блоками приёма (норма 16 мс)
static volatile int32_t wake_frame[2] = {-1, -1};         // первый непустой кадр линии после перезапуска
static volatile uint32_t stale[2] = {0, 0};               // старых буферов выброшено после перезапусков (по линиям)

static QueueHandle_t net_queue;                      // блоки звука для Wi-Fi (16 блоков = 256 мс на провалы связи)

// Запись «c»: цикл выделяет память и просит, задача приёма заполняет, цикл отдаёт по USB и освобождает.
enum { CAP_IDLE, CAP_RUN, CAP_DONE };
static volatile int cap_state = CAP_IDLE;
static int16_t *cap_buf = nullptr;
static int cap_filled = 0;
static uint32_t cap_started = 0;

enum Verdict : uint8_t { V_WAIT, V_OK, V_SPILL, V_GHOST, V_NONE, V_FLOAT, V_GAPS, V_NOISE, V_HOP };
static const char *const VERDICT[] = {"wait", "ok", "spill", "ghost", "none", "float", "gaps", "noise", "hop"};
static const char *const L_STATE[] = {"ok", "ok", "ok", "копия", "нет", "наводка", "ok", "мусор", "прыгает"};

struct Chan {
  uint8_t v = V_WAIT;
  float zero = 0, tail = 0, copy = 0, rms = -140, peak = -140, r1 = 0;   // окно ~0,5 с
  float frms = -140, fpeak = -140;                                        // окно ~0,2 с
  bool stuck = true;                                                      // все отсчёты окна ~0,2 с одинаковые
};
static Chan pub[4];
static volatile uint32_t fast_seq = 0;

static void route_shared_clock() {
  // I2S0 (ведущий, приём) выдаёт такт на 26/25; I2S1 (ведомый) читает его с тех же площадок.
  gpio_set_direction((gpio_num_t)PIN_SCK, GPIO_MODE_INPUT_OUTPUT);
  gpio_set_direction((gpio_num_t)PIN_WS, GPIO_MODE_INPUT_OUTPUT);
  gpio_matrix_out(PIN_SCK, I2S0I_BCK_OUT_IDX, false, false);
  gpio_matrix_out(PIN_WS, I2S0I_WS_OUT_IDX, false, false);
  gpio_matrix_in(PIN_SCK, I2S1I_BCK_IN_IDX, false);
  gpio_matrix_in(PIN_WS, I2S1I_WS_IN_IDX, false);
}

static void install(i2s_port_t port, bool master, int data_pin) {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)((master ? I2S_MODE_MASTER : I2S_MODE_SLAVE) | I2S_MODE_RX);
  cfg.sample_rate = RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;   // INMP441: 24 бита в 32-битном слоте
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = CHUNK;
  cfg.use_apll = false;
  i2s_driver_install(port, &cfg, 16, &events[port]);  // события: по ним видно переполнение буфера
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = PIN_SCK;
  pins.ws_io_num = PIN_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = data_pin;
  i2s_set_pin(port, &pins);
}

// Счётчики импульсов на ножках (команда d): аппаратный PCNT читает те же площадки через матрицу GPIO, не меняя ни
// направления, ни подтяжек — приём звука идёт как шёл. Фильтр 10 тактов APB (125 нс) гасит иголки наводок.
static const int COUNT_PINS[4] = {PIN_SCK, PIN_WS, PIN_SD1, PIN_SD2};
static void setup_counters() {
  static const uint32_t signals[4] = {PCNT_SIG_CH0_IN0_IDX, PCNT_SIG_CH0_IN1_IDX, PCNT_SIG_CH0_IN2_IDX,
                                      PCNT_SIG_CH0_IN3_IDX};
  for (int u = 0; u < 4; u++) {
    pcnt_config_t c = {};
    c.pulse_gpio_num = PCNT_PIN_NOT_USED;              // ножку подключаем сами: драйвер перевёл бы её во вход с подтяжкой
    c.ctrl_gpio_num = PCNT_PIN_NOT_USED;
    c.channel = PCNT_CHANNEL_0;
    c.unit = (pcnt_unit_t)u;
    c.pos_mode = PCNT_COUNT_INC;
    c.neg_mode = PCNT_COUNT_DIS;
    c.lctrl_mode = PCNT_MODE_KEEP;
    c.hctrl_mode = PCNT_MODE_KEEP;
    c.counter_h_lim = 32767;
    c.counter_l_lim = -32768;
    pcnt_unit_config(&c);
    pcnt_set_filter_value((pcnt_unit_t)u, 10);
    pcnt_filter_enable((pcnt_unit_t)u);
    gpio_matrix_in(COUNT_PINS[u], signals[u], false);
    pcnt_counter_pause((pcnt_unit_t)u);
    pcnt_counter_clear((pcnt_unit_t)u);
  }
}

// Строка D: частоты SCK/WS (Гц), фронты на линиях данных в секунду и доля «единиц» на каждой ножке.
// Исправный такт: WS 16000, SCK 1024000 (64 бита на кадр), оба ~50 %. Линия без микрофона: 0 фронтов, 0 %.
String pin_report() {
  static portMUX_TYPE count_mux = portMUX_INITIALIZER_UNLOCKED;
  for (int u = 0; u < 4; u++) { pcnt_counter_pause((pcnt_unit_t)u); pcnt_counter_clear((pcnt_unit_t)u); }
  portENTER_CRITICAL(&count_mux);
  int64_t t0 = esp_timer_get_time();
  for (int u = 0; u < 4; u++) pcnt_counter_resume((pcnt_unit_t)u);
  portEXIT_CRITICAL(&count_mux);
  vTaskDelay(pdMS_TO_TICKS(10));                       // ~10 мс: SCK ~10 тыс. фронтов, далеко до предела счётчика
  portENTER_CRITICAL(&count_mux);
  for (int u = 0; u < 4; u++) pcnt_counter_pause((pcnt_unit_t)u);
  int64_t dt = esp_timer_get_time() - t0;
  portEXIT_CRITICAL(&count_mux);
  uint32_t hz[4];
  for (int u = 0; u < 4; u++) {
    int16_t v = 0;
    pcnt_get_counter_value((pcnt_unit_t)u, &v);
    hz[u] = (uint32_t)((int64_t)(uint16_t)v * 1000000 / (dt > 0 ? dt : 1));
  }
  const int N = 20000;
  int high[4] = {0, 0, 0, 0};
  for (int i = 0; i < N; i++)
    for (int k = 0; k < 4; k++) high[k] += gpio_get_level((gpio_num_t)COUNT_PINS[k]);
  char line[200];
  snprintf(line, sizeof(line), "D SCK26 %lu %d%% | WS25 %lu %d%% | SD33 %lu %d%% | SD32 %lu %d%% | i2s_fail %lu | ovf %lu",
           (unsigned long)hz[0], high[0] * 100 / N, (unsigned long)hz[1], high[1] * 100 / N,
           (unsigned long)hz[2], high[2] * 100 / N, (unsigned long)hz[3], high[3] * 100 / N,
           (unsigned long)i2s_failures, (unsigned long)overflows);
  return String(line);
}

// Фильтр ниже 60 Гц (Баттерворт 2-го порядка, транспонированная прямая форма II): убирает дрейф и инфразвук.
static float hb0, hb1, hb2, ha1, ha2, hz1[4], hz2[4];
static void setup_highpass(double fc) {
  double w = 2 * M_PI * fc / RATE, c = cos(w), a = sin(w) / (2 * M_SQRT1_2), a0 = 1 + a;
  hb0 = (1 + c) / 2 / a0;
  hb1 = -(1 + c) / a0;
  hb2 = (1 + c) / 2 / a0;
  ha1 = -2 * c / a0;
  ha2 = (1 - a) / a0;
}

// Перезапуск обоих I2S с одного кадра. Недочитанное до остановки выбрасывается: иначе одна линия отдала бы
// старый блок, а другая новый — и они разъехались бы на 16 мс.
static void start_aligned() {
  i2s_stop(I2S_NUM_0);
  i2s_stop(I2S_NUM_1);
  for (int p = 0; p < 2; p++) {
    size_t got;
    do {
      got = 0;
      i2s_read((i2s_port_t)p, p ? buf1 : buf0, sizeof(buf0), &got, 0);
    } while (got);
    xQueueReset(events[p]);
  }
  i2s_zero_dma_buffer(I2S_NUM_0);
  i2s_zero_dma_buffer(I2S_NUM_1);
  for (int ch = 0; ch < 4; ch++) hz1[ch] = hz2[ch] = 0;
  wake_frame[0] = wake_frame[1] = -1;
  // Оба старта неразрывно: в записи из робота (ushi-11, после аварийных перезапусков) М1 отставал от М2/М3 на
  // ~340 кадров (21 мс) — похоже, между стартом ведомого и ведущего что-то вклинилось на 21 мс, а такт ведущего
  // после остановки, видимо, не замирает. Под критической секцией между двумя стартами ничего не встанет.
  static portMUX_TYPE start_mux = portMUX_INITIALIZER_UNLOCKED;
  portENTER_CRITICAL(&start_mux);
  i2s_start(I2S_NUM_1);                // ведомый ждёт такта
  i2s_start(I2S_NUM_0);                // такт пошёл
  portEXIT_CRITICAL(&start_mux);
  // 18.10, щелчки спереди: после перезапуска М1 (I2S0) отставал от М2/М3 ровно на блок (~256 кадров). Причина —
  // прерывание «блок готов», оставшееся от прошлого запуска: включившись, драйвер кладёт в очередь старый буфер,
  // и одна линия навсегда читает на блок раньше. Настоящий первый блок готов только через 16 мс — всё, что
  // попало в очереди за первые 3 мс, старое: выбросить (stale — сколько таких было).
  vTaskDelay(pdMS_TO_TICKS(3));
  for (int p = 0; p < 2; p++) {
    size_t got;
    do {
      got = 0;
      i2s_read((i2s_port_t)p, p ? buf1 : buf0, sizeof(buf0), &got, 0);
      if (got) stale[p]++;
    } while (got);
    xQueueReset(events[p]);
  }
  frames_total = 0;
  epoch++;
}

// Читает по CHUNK кадров с обоих I2S. false — ведомый не получил такт (нет SCK/WS или чужой провод на них).
static bool read_chunk() {
  size_t got0 = 0, got1 = 0;
  i2s_read(I2S_NUM_0, buf0, sizeof(buf0), &got0, pdMS_TO_TICKS(100));
  i2s_read(I2S_NUM_1, buf1, sizeof(buf1), &got1, pdMS_TO_TICKS(100));
  if (got0 != sizeof(buf0) || got1 != sizeof(buf1)) return false;
  frames_total += CHUNK;
  return true;
}

static bool overflowed() {
  bool ovf = false;
  i2s_event_t e;
  for (int p = 0; p < 2; p++)
    while (xQueueReceive(events[p], &e, 0) == pdTRUE) ovf |= e.type == I2S_EVENT_RX_Q_OVF;
  return ovf;
}

// Порядок слотов в буфере драйвера ESP32 для RIGHT_LEFT, 32 бита: [левый, правый] — проверено на стенде 15.10:
// микрофон с L/R прямо на земле отвечал во втором слоте, пока порядок был записан наоборот. М1 и М3 — левые (L/R на GND).
static inline int32_t word_of(int ch, int i) {
  switch (ch) {
    case 0: return buf0[2 * i];          // слот 0: SD1, левый
    case 1: return buf0[2 * i + 1];      // слот 1: SD1, правый
    case 2: return buf1[2 * i];          // слот 2: SD2, левый
    default: return buf1[2 * i + 1];     // слот 3: SD2, правый
  }
}
// Звук после фильтра → 16 бит: 24 бита → 18, насыщение в 16 (как всегда у записи «c» и звука по Wi-Fi).
static inline int16_t to16(float y) {
  float v = y * (1.f / 64.f);
  return (int16_t)(v > 32767.f ? 32767 : v < -32768.f ? -32768 : lrintf(v));
}

static void filter_chunk() {
  for (int i = 0; i < CHUNK; i++) {
    const int32_t w[4] = {buf0[2 * i], buf0[2 * i + 1], buf1[2 * i], buf1[2 * i + 1]};
    for (int ch = 0; ch < 4; ch++) {
      float x = (float)(w[ch] >> 8);
      float y = hb0 * x + hz1[ch];
      hz1[ch] = hb1 * x - ha1 * y + hz2[ch];
      hz2[ch] = hb2 * x - ha2 * y;
      hp[i][ch] = y;
    }
  }
}

// Слово слота, принятое I2S (замер 17.10, сырые слова «x» с живым микрофоном): биты 31..8 — 24 бита звука,
// биты 7..1 — нули (микрофон отпустил линию, подтяжка), бит 0 — старший бит СЛЕДУЮЩЕГО слота: соседний микрофон
// выставляет его на такт раньше, чем I2S начинает новое слово. Поэтому пустой слот рядом с живым микрофоном —
// это 00000000 или 00000001, а «наводка» — мусор в битах 5..1 (бит 7–6 могут держать младший бит звука, пока линия
// разряжается через подтяжку).
static inline bool empty_word(int32_t w) { return (w & ~1) == 0; }
static inline bool dirty_tail(int32_t w) { return (w & 0x3E) != 0; }

// Накопители окна. Суммы блока (256 отсчётов) считаются во float, окна копятся в double.
struct Acc {
  double sum = 0, sum2 = 0, lag = 0;
  float prev = 0, peak = 0;
  uint32_t n = 0, zero = 0, tail = 0;
};
static Acc slow_acc[4], fast_acc[4];
static uint32_t slow_copy[2] = {0, 0}, slow_either[2] = {0, 0}, slow_both[2] = {0, 0};
static int slow_chunks = 0, fast_chunks = 0, stuck_chunks[4] = {0, 0, 0, 0};

// Копия на линии: в правом слоте те же 24 бита, что в левом этого же, прошлого или следующего кадра (микрофон с
// болтающимся L/R отвечает в обоих слотах; 15.10 копия была в том же кадре, 17.10 — со сдвигом на кадр).
static inline uint32_t copies(const int32_t *b) {
  uint32_t n = 0;
  for (int i = 0; i < CHUNK; i++) {
    int32_t r = b[2 * i + 1] >> 8;
    if (r == 0) continue;
    n += r == b[2 * i] >> 8 || (i > 0 && r == b[2 * i - 2] >> 8) || (i + 1 < CHUNK && r == b[2 * i + 2] >> 8);
  }
  return n;
}

// На линии звучит хоть один слот / оба сразу (для «прыгает»: один микрофон по очереди в обоих слотах).
static inline void occupancy(const int32_t *b, uint32_t &either, uint32_t &both) {
  for (int i = 0; i < CHUNK; i++) {
    bool l = !empty_word(b[2 * i]), r = !empty_word(b[2 * i + 1]);
    either += l || r;
    both += l && r;
  }
}

static void level_of(const Acc &a, float &rms, float &peak, float *r1 = nullptr) {
  double n = a.n ? a.n : 1;
  double mean = a.sum / n;
  double var = a.sum2 / n - mean * mean;
  rms = var > 0 ? 10. * log10(var / 70368744177664.) : -140.;   // полная шкала 24 бит: (2^23)^2
  peak = a.peak > 0 ? 20. * log10(a.peak / 8388608.) : -140.;
  if (r1) *r1 = var > 1. ? (a.lag / n - mean * mean) / var : 0.;
}

// either/both — доли кадров, где на линии звучит хоть один слот / оба сразу.
static uint8_t verdict(const Chan &c, const Chan &pair, bool right, float either, float both) {
  if (c.zero >= .97f) return V_NONE;
  if (c.tail >= .20f) return V_FLOAT;
  if (c.zero >= .15f && c.zero <= .85f && pair.zero >= .15f && pair.zero <= .85f && either >= .9f && both <= .15f)
    return V_HOP;
  if (c.copy >= .10f) return c.zero > pair.zero || (c.zero == pair.zero && right) ? V_GHOST : V_SPILL;
  if (c.zero >= .10f) return V_GAPS;
  if (c.rms > -20.f && c.r1 < .3f) return V_NOISE;
  return V_OK;
}

static void account() {
  bool same[4] = {true, true, true, true};
  int32_t first[4];
  for (int ch = 0; ch < 4; ch++) first[ch] = word_of(ch, 0);
  for (int ch = 0; ch < 4; ch++) {
    float s1 = 0, s2 = 0, sl = 0, pk = 0, prev = slow_acc[ch].prev;
    uint32_t zero = 0, tail = 0;
    for (int i = 0; i < CHUNK; i++) {
      float y = hp[i][ch];
      s1 += y;
      s2 += y * y;
      sl += y * prev;
      prev = y;
      float m = fabsf(y);
      if (m > pk) pk = m;
      int32_t w = word_of(ch, i);
      zero += empty_word(w);
      tail += dirty_tail(w);
      same[ch] &= w == first[ch];
    }
    for (Acc *a : {&slow_acc[ch], &fast_acc[ch]}) {
      a->sum += s1;
      a->sum2 += s2;
      a->lag += sl;
      a->prev = prev;
      if (pk > a->peak) a->peak = pk;
      a->n += CHUNK;
      a->zero += zero;
      a->tail += tail;
    }
  }
  slow_copy[0] += copies(buf0);
  slow_copy[1] += copies(buf1);
  occupancy(buf0, slow_either[0], slow_both[0]);
  occupancy(buf1, slow_either[1], slow_both[1]);
  for (int ch = 0; ch < 4; ch++) stuck_chunks[ch] += same[ch];
  // Первый непустой кадр каждой линии после перезапуска (микрофоны после остановки такта молчат, пока не проснутся).
  for (int line = 0; line < 2; line++)
    if (wake_frame[line] < 0)
      for (int i = 0; i < CHUNK; i++)
        if (!empty_word(word_of(2 * line, i)) || !empty_word(word_of(2 * line + 1, i))) {
          wake_frame[line] = frames_total - CHUNK + i;
          break;
        }
  if (++fast_chunks >= FAST_CHUNKS) {
    Chan f[4];
    for (int ch = 0; ch < 4; ch++) {
      level_of(fast_acc[ch], f[ch].frms, f[ch].fpeak);
      f[ch].stuck = stuck_chunks[ch] >= fast_chunks;
      fast_acc[ch] = Acc();
      fast_acc[ch].prev = slow_acc[ch].prev;
      stuck_chunks[ch] = 0;
    }
    fast_chunks = 0;
    portENTER_CRITICAL(&mux);
    for (int ch = 0; ch < 4; ch++) { pub[ch].frms = f[ch].frms; pub[ch].fpeak = f[ch].fpeak; pub[ch].stuck = f[ch].stuck; }
    fast_seq++;
    portEXIT_CRITICAL(&mux);
  }
  if (++slow_chunks >= SLOW_CHUNKS) {
    Chan s[4];
    for (int ch = 0; ch < 4; ch++) {
      const Acc &a = slow_acc[ch];
      float n = a.n ? a.n : 1;
      s[ch].zero = a.zero / n;
      s[ch].tail = a.tail / n;
      s[ch].copy = slow_copy[ch / 2] / n;
      level_of(a, s[ch].rms, s[ch].peak, &s[ch].r1);
    }
    for (int ch = 0; ch < 4; ch++) {
      float n = slow_acc[ch].n ? slow_acc[ch].n : 1;
      s[ch].v = verdict(s[ch], s[ch ^ 1], ch & 1, slow_either[ch / 2] / n, slow_both[ch / 2] / n);
    }
    for (int ch = 0; ch < 4; ch++) {
      float prev = slow_acc[ch].prev;
      slow_acc[ch] = Acc();
      slow_acc[ch].prev = prev;
    }
    slow_copy[0] = slow_copy[1] = slow_either[0] = slow_either[1] = slow_both[0] = slow_both[1] = 0;
    slow_chunks = 0;
    portENTER_CRITICAL(&mux);
    for (int ch = 0; ch < 4; ch++) {
      pub[ch].v = s[ch].v; pub[ch].zero = s[ch].zero; pub[ch].tail = s[ch].tail; pub[ch].copy = s[ch].copy;
      pub[ch].rms = s[ch].rms; pub[ch].peak = s[ch].peak; pub[ch].r1 = s[ch].r1;
    }
    portEXIT_CRITICAL(&mux);
  }
}

static void capture_fill() {
  if (cap_state != CAP_RUN || cap_buf == nullptr) return;
  for (int i = 0; i < CHUNK && cap_filled < CAPTURE; i++, cap_filled++)
    for (int ch = 0; ch < 4; ch++) cap_buf[cap_filled * 4 + ch] = to16(hp[i][ch]);
  if (cap_filled >= CAPTURE) cap_state = CAP_DONE;
}

static void stream_push() {
  if (!net_streaming()) return;
  static AudioChunk c, junk;
  c.first = frames_total - CHUNK;
  c.epoch = epoch;
  for (int i = 0; i < CHUNK; i++)
    for (int ch = 0; ch < 4; ch++) c.s[i * 4 + ch] = to16(hp[i][ch]);
  if (xQueueSend(net_queue, &c, 0) != pdTRUE) {          // Wi-Fi не успевает: выбросить самый старый блок
    xQueueReceive(net_queue, &junk, 0);
    queue_drops++;
    xQueueSend(net_queue, &c, 0);
  }
}

// Сырые слова (USB «x», UDP «raw»): 8 кадров подряд × 4 слота в шестнадцатеричном виде, как их принял I2S.
// Для разбора «что на линии»: у живого INMP441 младший байт слова — нули (линия отпущена, подтяжка).
static volatile bool raw_req = false, raw_ready = false;
static int32_t raw_words[8][4];

static void raw_grab() {
  if (!raw_req) return;
  for (int i = 0; i < 8; i++)
    for (int ch = 0; ch < 4; ch++) raw_words[i][ch] = word_of(ch, i);
  raw_req = false;
  raw_ready = true;
}

String raw_dump() {
  raw_ready = false;
  raw_req = true;
  for (int i = 0; i < 20 && !raw_ready; i++) vTaskDelay(pdMS_TO_TICKS(5));
  if (!raw_ready) return String("X нет данных");
  String out = "X";
  char cell[12];
  for (int i = 0; i < 8; i++) {
    out += " |";
    for (int ch = 0; ch < 4; ch++) {
      snprintf(cell, sizeof(cell), " %08lx", (unsigned long)(uint32_t)raw_words[i][ch]);
      out += cell;
    }
  }
  return out;
}

void audio_restart() { restart_req = true; }

static void audio_task(void *) {
  bool running = false;
  uint32_t last_us = 0;
  for (;;) {
    if (paused) {                                        // идёт приём прошивки по Wi-Fi
      if (running) { i2s_stop(I2S_NUM_0); i2s_stop(I2S_NUM_1); running = false; }
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }
    if (!running || restart_req) {
      restart_req = false;
      start_aligned();
      running = true;
      last_us = 0;
    }
    if (!read_chunk()) {                                 // такт пропал: одна линия могла отдать блок, другая нет
      i2s_failures++;
      restart_req = true;
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }
    uint32_t now = micros();
    if (last_us) {
      uint32_t gap = now - last_us;
      if (gap > gap_max_us) gap_max_us = gap;
      if (gap > 40000) stalls++;
    }
    last_us = now;
    if (overflowed()) {                                  // буфер переполнился: линии могли разъехаться — заново
      overflows++;
      restart_req = true;
      continue;
    }
    filter_chunk();
    account();
    raw_grab();
    capture_fill();
    stream_push();
  }
}

// Строка H (USB «h», UDP «health»): по каналу — вывод, доли пустых слов / «хвостов» / копий (%), уровни после фильтра
// (дБ 24 бит), похожесть соседних отсчётов; счётчики переполнений, сбоев такта, эпох, выброшенных блоков; первый звук
// линий после перезапуска (wake0/wake1, кадр); самая длинная пауза приёма (gap, мс, с прошлого запроса) и паузы
// длиннее 40 мс (stall).
String health_json() {
  Chan c[4];
  portENTER_CRITICAL(&mux);
  for (int ch = 0; ch < 4; ch++) c[ch] = pub[ch];
  portEXIT_CRITICAL(&mux);
  char line[900];
  int k = snprintf(line, sizeof(line), "{\"fw\":\"%s\",\"v\":[\"%s\",\"%s\",\"%s\",\"%s\"]", FW_VERSION,
                   VERDICT[c[0].v], VERDICT[c[1].v], VERDICT[c[2].v], VERDICT[c[3].v]);
  auto arr = [&](const char *name, float Chan::*field, float scale, const char *fmt) {
    k += snprintf(line + k, sizeof(line) - k, ",\"%s\":[", name);
    for (int ch = 0; ch < 4; ch++)
      k += snprintf(line + k, sizeof(line) - k, fmt, c[ch].*field * scale, ch < 3 ? "," : "]");
  };
  arr("zero", &Chan::zero, 100.f, "%.1f%s");
  arr("tail", &Chan::tail, 100.f, "%.1f%s");
  arr("copy", &Chan::copy, 100.f, "%.1f%s");
  arr("rms", &Chan::rms, 1.f, "%.1f%s");
  arr("peak", &Chan::peak, 1.f, "%.1f%s");
  arr("frms", &Chan::frms, 1.f, "%.1f%s");
  arr("fpeak", &Chan::fpeak, 1.f, "%.1f%s");
  arr("r1", &Chan::r1, 1.f, "%.2f%s");
  uint32_t gap = gap_max_us;
  gap_max_us = 0;
  snprintf(line + k, sizeof(line) - k,
           ",\"ovf\":%lu,\"fail\":%lu,\"epoch\":%lu,\"qdrop\":%lu,\"frames\":%lu,"
           "\"wake0\":%ld,\"wake1\":%ld,\"gap\":%.1f,\"stall\":%lu,\"vslot\":%d,\"stale\":[%lu,%lu]}",
           (unsigned long)overflows, (unsigned long)i2s_failures, (unsigned long)epoch, (unsigned long)queue_drops,
           (unsigned long)frames_total, (long)wake_frame[0],
           (long)wake_frame[1], gap / 1000.f, (unsigned long)stalls, voice_slot_auto(), (unsigned long)stale[0],
           (unsigned long)stale[1]);
  return String(line);
}

String audio_counters() {
  char line[120];
  snprintf(line, sizeof(line), "\"ovf\":%lu,\"fail\":%lu,\"epoch\":%lu,\"qdrop\":%lu,\"stall\":%lu",
           (unsigned long)overflows, (unsigned long)i2s_failures, (unsigned long)epoch, (unsigned long)queue_drops,
           (unsigned long)stalls);
  return String(line);
}

// Слот для голоса «по умолчанию»: живой микрофон, громче всех за последние 0,2 с; переключение — если другой громче
// на 3 дБ хотя бы полсекунды (без дёрганья посреди фразы).
int voice_slot_auto() {
  static int current = 0;
  static uint32_t better_since = 0;
  Chan c[4];
  portENTER_CRITICAL(&mux);
  for (int ch = 0; ch < 4; ch++) c[ch] = pub[ch];
  portEXIT_CRITICAL(&mux);
  auto alive = [&](int ch) { return c[ch].v == V_OK || c[ch].v == V_SPILL || c[ch].v == V_GAPS; };
  int best = -1;
  for (int ch = 0; ch < 4; ch++)
    if (alive(ch) && (best < 0 || c[ch].frms > c[best].frms)) best = ch;
  if (best < 0) return current;
  if (!alive(current)) { current = best; better_since = 0; return current; }
  if (best != current && c[best].frms > c[current].frms + 3.f) {
    if (!better_since) better_since = millis();
    else if (millis() - better_since > 500) { current = best; better_since = 0; }
  } else {
    better_since = 0;
  }
  return current;
}

// Строка L (5 раз в секунду): дБ полной шкалы 24 бит и состояние: «ok», «нет» (все отсчёты окна одинаковые),
// «копия», «наводка», «мусор», «прыгает» (по последнему выводу самопроверки).
static void print_levels() {
  static uint32_t seen = 0;
  if (fast_seq == seen) return;
  Chan c[4];
  portENTER_CRITICAL(&mux);
  seen = fast_seq;
  for (int ch = 0; ch < 4; ch++) c[ch] = pub[ch];
  portEXIT_CRITICAL(&mux);
  Serial.printf("L");
  for (int ch = 0; ch < 4; ch++)
    Serial.printf(" M%d %6.1f %6.1f %s", ch + 1, c[ch].frms, c[ch].fpeak, c[ch].stuck ? "нет" : L_STATE[c[ch].v]);
  Serial.printf("\n");
}

static void capture_start() {
  if (cap_state != CAP_IDLE) return;
  cap_buf = (int16_t *)malloc(CAPTURE * 4 * sizeof(int16_t));
  if (cap_buf == nullptr) { Serial.println("E нет памяти"); return; }
  cap_filled = 0;
  cap_started = millis();
  cap_state = CAP_RUN;
}

static void capture_finish() {
  if (cap_state == CAP_RUN && millis() - cap_started > 3000) {
    cap_state = CAP_IDLE;
    free(cap_buf);
    cap_buf = nullptr;
    Serial.println("E нет такта на I2S1 или I2S0 (проверь SCK/WS)");
    return;
  }
  if (cap_state != CAP_DONE) return;
  // Блок: строка «CAP <кадров> <каналов> <частота>», затем кадры × каналы int16 LE, затем «END».
  Serial.printf("CAP %d 4 %d\n", CAPTURE, RATE);
  Serial.write((const uint8_t *)cap_buf, CAPTURE * 4 * sizeof(int16_t));
  Serial.printf("\nEND\n");
  free(cap_buf);
  cap_buf = nullptr;
  cap_state = CAP_IDLE;
}

static char mode = 'l';

void setup() {
  Serial.begin(921600);
  delay(300);
  setup_highpass(60.);
  install(I2S_NUM_1, false, PIN_SD2);
  install(I2S_NUM_0, true, PIN_SD1);
  route_shared_clock();
  // Линия данных без микрофона висит в воздухе и ловит наводки от такта (15.10: «мусор» на пустом столбике 12).
  // Подтяжка к земле (INMP441 тоже советует резистор на SD): пустая линия — чистые нули, микрофон её перебивает.
  gpio_pulldown_en((gpio_num_t)PIN_SD1);
  gpio_pulldown_en((gpio_num_t)PIN_SD2);
  setup_counters();
  net_queue = xQueueCreate(16, sizeof(AudioChunk));
  xTaskCreatePinnedToCore(audio_task, "audio", 4096, nullptr, 22, nullptr, 1);
  net_setup([]() { paused = true; }, []() { paused = false; });
  ble_setup();                                            // звук и по Bluetooth — телефону в корпусе (ushi-15)
  Serial.println("I " FW_VERSION " · 4×INMP441 · I2S0 ведущий SD1=33, I2S1 ведомый SD2=32, такт 26/25 · 16 кГц");
}

void loop() {
  net_tick();
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'l' || c == 's') mode = c;
    else if (c == 'c') capture_start();
    else if (c == 'r') { restart_req = true; Serial.println("I перезапуск I2S"); }
    else if (c == 'i') Serial.println("I " + net_status());
    else if (c == 'h') Serial.println("H " + health_json());
    else if (c == 'w') net_forget_wifi();
    else if (c == 'p') net_start_wps();
    else if (c == 'd') Serial.println(pin_report());
    else if (c == 'x') Serial.println(raw_dump());
  }
  capture_finish();
  if (ble_wedged()) {                                     // Bluetooth не отвечает ни на что (ushi-24): начать заново
    Serial.println("E Bluetooth завис: перезапуск платы");
    delay(100);
    ESP.restart();
  }
  static uint32_t warned_fail = 0, warned_at = 0;
  if (i2s_failures != warned_fail && millis() - warned_at > 2000) {
    warned_fail = i2s_failures;
    warned_at = millis();
    Serial.println("E нет данных от I2S (такт/проводка)");
  }
  if (mode == 'l') print_levels();
  static AudioChunk out;
  if (xQueueReceive(net_queue, &out, pdMS_TO_TICKS(5)) == pdTRUE) {
    net_audio(out);
    while (xQueueReceive(net_queue, &out, 0) == pdTRUE) net_audio(out);
  }
}
