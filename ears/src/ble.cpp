// Звук по Bluetooth LE (ushi-15, 18.10). Владелец: «подружи телефон с платой напрямую — они всегда рядом».
//
// Плата на дне корпуса, её Wi-Fi упал до −85…−93 дБм и терял 70 % звука; телефон — в 30 см. Плата объявляет себя
// «Bratan-ushi» с сервисом 6b9c0001-…; телефон (приложение лица, BleEars.java) подписывается на уведомления
// характеристики звука 6b9c0002-…, и плата шлёт туда те же пакеты BRM1 (три микрофона ADPCM, 32 мс), что и по Wi-Fi.
// Пакет режется по MTU: первый байт куска — бит 7 «здесь начало пакета» (дальше длина u16 LE), биты 0–6 — счётчик
// кусков; потерянный кусок губит один пакет, а не поток. Wi-Fi остаётся: прошивка по воздуху, «ping», панель и
// запасной путь звука, когда телефона по Bluetooth нет.
#include "ble.h"

#include <NimBLEDevice.h>
#include <WiFi.h>

static const char *SERVICE_UUID = "6b9c0001-3f6e-4f4b-9a54-6272617461e1";
static const char *AUDIO_UUID = "6b9c0002-3f6e-4f4b-9a54-6272617461e1";
// ushi-19 (18.10): the microphones went to anyone in range who subscribed — now only to the connection that wrote the
// token (built in from work/ushi/.ble-token, the face app writes the same); a wrong one is disconnected at once.
static const char *UNLOCK_UUID = "6b9c0003-3f6e-4f4b-9a54-6272617461e1";
#ifndef BLE_TOKEN
#define BLE_TOKEN ""
#endif

static NimBLECharacteristic *audio = nullptr;
static volatile bool subscribed = false;
static volatile uint16_t sub_conn = BLE_HS_CONN_HANDLE_NONE, unlocked = BLE_HS_CONN_HANDLE_NONE;
// Who is subscribed, per connection: a stranger's subscription must not take the stream's place from the phone.
static const int SUBS = 4;
static volatile uint16_t subs[SUBS] = {BLE_HS_CONN_HANDLE_NONE, BLE_HS_CONN_HANDLE_NONE, BLE_HS_CONN_HANDLE_NONE,
                                       BLE_HS_CONN_HANDLE_NONE};

static bool is_subscribed(uint16_t handle) {
  for (int i = 0; i < SUBS; i++) if (subs[i] == handle && handle != BLE_HS_CONN_HANDLE_NONE) return true;
  return false;
}

static void set_subscribed(uint16_t handle, bool on) {
  for (int i = 0; i < SUBS; i++) if (subs[i] == handle) subs[i] = BLE_HS_CONN_HANDLE_NONE;
  if (!on) return;
  for (int i = 0; i < SUBS; i++) if (subs[i] == BLE_HS_CONN_HANDLE_NONE) { subs[i] = handle; return; }
}

static void choose_stream() {                           // the unlocked connection, if it is subscribed — no one else
  sub_conn = is_subscribed(unlocked) ? unlocked : BLE_HS_CONN_HANDLE_NONE;
}
static uint32_t refused = 0;
static volatile uint16_t peer_mtu = 23;
static volatile uint16_t peer = BLE_HS_CONN_HANDLE_NONE;
static uint8_t counter = 0;
static int dle_rc = -1;
static uint32_t sent_chunks = 0, failed_chunks = 0, connects = 0, waits = 0;
static void start_sender();
static const int WAIT_MS = 40;                          // сколько ждать свободных буферов на один кусок

// ushi-24 (18.10 22:17): после 2,3 ч и 77 подключений плата принимала связь, но ни на один запрос по ней не отвечала
// (ни телефону, ни ПК: «Could not get GATT services: Unreachable»), мост молчал. Счётчики стека — в «ping»: свободные
// буферы (mbufs) и места в контроллере под пакеты (acl); связь без единого обмена — «немая». Шесть немых подряд —
// стек завис: плата перезапускается (не раньше 10 мин после старта; робот переживает перезапуск платы, C52).
extern "C" {
extern uint16_t ble_hs_hci_avail_pkts;
int os_msys_num_free(void);
}
static volatile bool exchanged = false;                // на последней связи был хоть один обмен (MTU, подписка, запись)
static volatile uint32_t silent_links = 0, silent_total = 0;
static const uint32_t SILENT_LIMIT = 6;
static const uint32_t SILENT_REBOOT_AFTER_MS = 600000;

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer *server, ble_gap_conn_desc *desc) override {
    connects++;
    exchanged = false;
    // Короткий интервал связи: 31 пакет в секунду по 2 куска.
    server->updateConnParams(desc->conn_handle, 6, 12, 0, 200);
    // Длинные радиопакеты (Data Length Extension, 251 байт вместо 27): без них кусок в 512 байт уходит двадцатью
    // пакетами, и с телефона мост пропускал лишь треть звука (ushi-17: потери 64 %).
    dle_rc = ble_gap_set_data_len(desc->conn_handle, 251, 2120);
    // Still seen while connected (ushi-20): a link the phone left hanging must not hide the board from its next try
    // (18.10, the face app reinstalled: discovery never answered, and nobody could connect again but that phone).
    NimBLEDevice::startAdvertising();
  }
  void onDisconnect(NimBLEServer *server, ble_gap_conn_desc *desc) override {
    if (exchanged) silent_links = 0;
    else { silent_links++; silent_total++; }
    set_subscribed(desc->conn_handle, false);
    if (desc->conn_handle == unlocked) unlocked = BLE_HS_CONN_HANDLE_NONE;
    choose_stream();
    subscribed = sub_conn != BLE_HS_CONN_HANDLE_NONE;
    NimBLEDevice::startAdvertising();
  }
  void onMTUChange(uint16_t mtu, ble_gap_conn_desc *desc) override {
    exchanged = true;
    peer_mtu = mtu;
  }
};

class AudioCallbacks : public NimBLECharacteristicCallbacks {
  void onSubscribe(NimBLECharacteristic *c, ble_gap_conn_desc *desc, uint16_t value) override {
    exchanged = true;
    set_subscribed(desc->conn_handle, (value & 1) != 0);
    choose_stream();
    if (sub_conn != BLE_HS_CONN_HANDLE_NONE) peer_mtu = NimBLEDevice::getServer()->getPeerMTU(sub_conn);
    peer = sub_conn;
    subscribed = sub_conn != BLE_HS_CONN_HANDLE_NONE;
  }
};

class UnlockCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic *c, ble_gap_conn_desc *desc) override {
    exchanged = true;
    std::string value = c->getValue();
    const std::string token = BLE_TOKEN;
    c->setValue("");                                    // never kept readable
    if (!token.empty() && value == token) {
      unlocked = desc->conn_handle;
    } else {
      refused++;
      NimBLEDevice::getServer()->disconnect(desc->conn_handle);
    }
    choose_stream();
    if (sub_conn != BLE_HS_CONN_HANDLE_NONE) peer_mtu = NimBLEDevice::getServer()->getPeerMTU(sub_conn);
    peer = sub_conn;
    subscribed = sub_conn != BLE_HS_CONN_HANDLE_NONE;
  }
};

void ble_setup() {
  // Wi-Fi и Bluetooth делят одно радио: при обоих ESP-IDF требует экономию модема у Wi-Fi (без неё — отказ).
  WiFi.setSleep(true);
  NimBLEDevice::init("Bratan-ushi");
  NimBLEDevice::setMTU(517);
  NimBLEDevice::setPower(ESP_PWR_LVL_N0);              // телефон в 30 см: громче не нужно, Wi-Fi в том же радио
  NimBLEServer *server = NimBLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  NimBLEService *service = server->createService(SERVICE_UUID);
  audio = service->createCharacteristic(AUDIO_UUID, NIMBLE_PROPERTY::NOTIFY);
  audio->setCallbacks(new AudioCallbacks());
  NimBLECharacteristic *unlock = service->createCharacteristic(UNLOCK_UUID, NIMBLE_PROPERTY::WRITE);
  unlock->setCallbacks(new UnlockCallbacks());
  service->start();
  NimBLEAdvertising *adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  adv->start();
  start_sender();
}

bool ble_streaming() { return subscribed && audio != nullptr; }

// Уведомление с проверкой: в NimBLE 1.4 notify() молча теряет кусок, когда кончились буферы (C52, ПК: 4.5 пакета/с
// вместо 31, недособранных 203 из 272). Здесь — ждём буфер и повторяем, пока кусок не уйдёт.
static bool notify_chunk(const uint8_t *data, size_t len) {
  for (int waited = 0; waited <= WAIT_MS; waited++) {
    if (!subscribed) return false;
    os_mbuf *om = ble_hs_mbuf_from_flat(data, len);
    if (om != nullptr) {
      int rc = ble_gattc_notify_custom(peer, audio->getHandle(), om);   // om освобождается и при отказе
      if (rc == 0) return true;
      if (rc != BLE_HS_ENOMEM && rc != BLE_HS_EBUSY) return false;
    }
    waits++;
    vTaskDelay(1);
  }
  return false;
}

// Отправка — в своей задаче с очередью на 4 пакета: ожидание буферов Bluetooth не должно держать loop() (там Wi-Fi
// роботу и сторож loop: ushi-16 первой сборки перезагружался сторожем через 7 с, сброс 6 = TASK_WDT).
static void send_packet(const uint8_t *packet, size_t len) {
  size_t room = peer_mtu > 3 ? peer_mtu - 3 : 20;      // полезная часть уведомления
  if (room > 509) room = 509;
  static uint8_t chunk[512];
  size_t pos = 0;
  bool first = true;
  while (pos < len) {
    size_t head = first ? 3 : 1;
    size_t n = len - pos < room - head ? len - pos : room - head;
    chunk[0] = (first ? 0x80 : 0) | (counter++ & 0x7F);
    if (first) { chunk[1] = len & 0xFF; chunk[2] = len >> 8; }
    memcpy(chunk + head, packet + pos, n);
    if (!notify_chunk(chunk, head + n)) {              // буферов нет и за 40 мс: остаток пакета бросаем, приёмник
      failed_chunks++;                                 // увидит разрыв по счётчику кусков
      return;
    }
    sent_chunks++;
    pos += n;
    first = false;
  }
}

static const size_t PACKET_MAX = 1024;                 // BRM1 трёх микрофонов — 800 байт
struct Packet {
  uint16_t len;
  uint8_t data[PACKET_MAX];
};
static QueueHandle_t queue = nullptr;
static uint32_t queue_drops = 0;

static void sender_task(void *) {
  static Packet p;
  for (;;) {
    if (xQueueReceive(queue, &p, portMAX_DELAY) == pdTRUE && subscribed) send_packet(p.data, p.len);
  }
}

static void start_sender() {
  queue = xQueueCreate(4, sizeof(Packet));
  xTaskCreatePinnedToCore(sender_task, "ble_send", 3072, nullptr, 2, nullptr, 1);
}

void ble_send(const uint8_t *packet, size_t len) {
  if (!ble_streaming() || queue == nullptr) return;
  static Packet p;
  if (len > PACKET_MAX) { queue_drops++; return; }
  p.len = len;
  memcpy(p.data, packet, len);
  if (xQueueSend(queue, &p, 0) != pdTRUE) queue_drops++;   // Bluetooth не успевает: новый пакет бросаем, Wi-Fi не ждёт
}

String ble_status() {
  ble_gap_conn_desc desc;
  int itvl = (subscribed && ble_gap_conn_find(peer, &desc) == 0) ? desc.conn_itvl * 125 / 100 : 0;   // мс
  char line[340];
  snprintf(line, sizeof(line),
           "\"ble\":{\"sub\":%s,\"mtu\":%u,\"itvl_ms\":%d,\"dle\":%d,\"chunks\":%lu,\"failed\":%lu,\"waits\":%lu,"
           "\"qdrops\":%lu,\"connects\":%lu,\"refused\":%lu,\"mbufs\":%d,\"acl\":%u,\"silent\":%lu,\"silent_total\":%lu}",
           subscribed ? "true" : "false", peer_mtu, itvl, dle_rc, (unsigned long)sent_chunks, (unsigned long)failed_chunks,
           (unsigned long)waits, (unsigned long)queue_drops, (unsigned long)connects, (unsigned long)refused,
           os_msys_num_free(), (unsigned)ble_hs_hci_avail_pkts, (unsigned long)silent_links,
           (unsigned long)silent_total);
  return String(line);
}

bool ble_wedged() { return silent_links >= SILENT_LIMIT && millis() > SILENT_REBOOT_AFTER_MS; }
