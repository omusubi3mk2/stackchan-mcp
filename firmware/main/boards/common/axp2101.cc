#include "axp2101.h"
#include "board.h"
#include "display.h"

#include <esp_log.h>

#define TAG "Axp2101"

Axp2101::Axp2101(i2c_master_bus_handle_t i2c_bus, uint8_t addr) : I2cDevice(i2c_bus, addr) {
}

int Axp2101::GetBatteryCurrentDirection() {
    return (ReadReg(0x01) & 0b01100000) >> 5;
}

bool Axp2101::IsCharging() {
    return GetBatteryCurrentDirection() == 1;
}

bool Axp2101::IsDischarging() {
    return GetBatteryCurrentDirection() == 2;
}

bool Axp2101::IsChargingDone() {
    uint8_t value = ReadReg(0x01);
    return (value & 0b00000111) == 0b00000100;
}

int Axp2101::GetBatteryLevel() {
    return ReadReg(0xA4);
}

float Axp2101::GetTemperature() {
    return ReadReg(0xA5);
}

void Axp2101::PowerOff() {
    uint8_t value = ReadReg(0x10);
    value = value | 0x01;
    WriteReg(0x10, value);
}

uint8_t Axp2101::ReadRegister(uint8_t reg) {
    return ReadReg(reg);
}

void Axp2101::ReadRegisters(uint8_t reg, uint8_t* buffer, size_t length) {
    ReadRegs(reg, buffer, length);
}

// TEMPORARY DEBUG: dump 0x00-0x9F to compare register state across a
// USB-less power-on-button failure cycle. Remove once root cause is found.
void Axp2101::DumpDebugRegisters(const char* tag) {
    ESP_LOGI(TAG, "==== AXP2101 register dump [%s] ====", tag);
    char line[3 * 16 + 1];
    for (int base = 0x00; base <= 0x90; base += 16) {
        int pos = 0;
        for (int i = 0; i < 16; i++) {
            uint8_t reg = base + i;
            uint8_t value = ReadReg(reg);
            pos += snprintf(line + pos, sizeof(line) - pos, "%02x ", value);
        }
        ESP_LOGI(TAG, "%02x: %s", base, line);
    }
    ESP_LOGI(TAG, "==== end dump [%s] ====", tag);
}
