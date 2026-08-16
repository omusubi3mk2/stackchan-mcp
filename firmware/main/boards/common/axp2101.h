#ifndef __AXP2101_H__
#define __AXP2101_H__

#include "i2c_device.h"

class Axp2101 : public I2cDevice {
public:
    Axp2101(i2c_master_bus_handle_t i2c_bus, uint8_t addr);
    bool IsCharging();
    bool IsDischarging();
    bool IsChargingDone();
    int GetBatteryLevel();
    float GetTemperature();
    void PowerOff();
    void DumpDebugRegisters(const char* tag);
    // On-demand register access for the self.pmic.read_reg MCP tool (WiFi/MCP
    // path, sidesteps the USB-serial observability gap that blocks debugging
    // USB-less boot failures — see stackchan_troubleshooting notes).
    uint8_t ReadRegister(uint8_t reg);
    void ReadRegisters(uint8_t reg, uint8_t* buffer, size_t length);

private:
    int GetBatteryCurrentDirection();
};

#endif
