// Звук по Bluetooth LE для телефона в том же корпусе (ushi-15, 18.10): см. ble.cpp.
#pragma once
#include <Arduino.h>

void ble_setup();
bool ble_streaming();                                  // телефон подписан на звук
void ble_send(const uint8_t *packet, size_t len);      // целый пакет BRM1 — кусками по MTU
String ble_status();                                    // для «ping»: подписан ли, MTU, куски, отказы
bool ble_wedged();                                      // стек завис: связи есть, обменов нет (ushi-24) — перезапуск
