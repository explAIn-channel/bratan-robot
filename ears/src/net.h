// Сеть «ушей»: Wi-Fi, который сам поднимается после любых сбоев, прошивка по воздуху с откатом, запросы по UDP 5771.
#pragma once
#include <Arduino.h>

#define FW_VERSION "ushi-24"

static const int CHUNK = 256;                 // кадров в блоке приёма (16 мс)

// Блок звука от задачи приёма к циклу: 4 слота int16 после фильтра, номер первого кадра, эпоха (перезапуск I2S).
struct AudioChunk {
  uint32_t first;
  uint32_t epoch;
  int16_t s[CHUNK * 4];
};

// on_ota_start: остановить микрофоны (идёт приём прошивки); on_ota_fail: запустить их снова.
void net_setup(void (*on_ota_start)(), void (*on_ota_fail)());
void net_tick();
bool net_online();
String net_status();          // одна строка JSON для «i» по USB и для «ping»
void net_forget_wifi();       // команда «w»: забыть сеть, снова точка Bratan-Ushi
void net_start_wps();         // команда «p»: подключиться по кнопке WPS роутера, без пароля
bool net_streaming();         // кто-то подписан на звук (sub/subv/suba на UDP 5771 не реже раза в 3 с)
void net_audio(const AudioChunk &c);   // разослать блок подписчикам (сырые 4 канала или голос одним каналом)

// Из main.cpp — ответы на запросы по UDP.
String health_json();         // «health»: что в каждом из 4 слотов, сдвиг линий, паузы приёма
String pin_report();          // «pins»: такты и линии данных
String audio_counters();      // счётчики приёма звука для «ping»
String raw_dump();            // «raw»: сырые 32-битные слова 8 кадров
int voice_slot_auto();        // слот для голоса «сам выбери»: живой и самый громкий
void audio_restart();         // «restart»: перезапустить оба I2S
