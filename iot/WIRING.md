# SignTalk IoT wiring (everything is on the ESP32)

The Raspberry Pi needs **no extra wiring**: it only runs the model, the web server and the MQTT broker.
The ESP32 holds the OLED, three LEDs, the buzzer and the emergency button, and talks to the Pi over Wi-Fi (MQTT).
Unplug the ESP32 while you wire.

| Part | ESP32 pin | Connection |
|---|---|---|
| OLED VCC | 3V3 | |
| OLED GND | GND | |
| OLED SDA | GPIO21 | |
| OLED SCL | GPIO22 | |
| Green LED (phrase recognised) | GPIO25 | GPIO25 -> 330 ohm resistor -> LED long leg (+); short leg (-) -> GND |
| Amber LED (sign not accepted) | GPIO26 | same, through its own 330 ohm resistor |
| Red LED (alert / Pi offline) | GPIO27 | same, through its own 330 ohm resistor |
| Buzzer | GPIO32 | see below |
| Emergency button | GPIO33 | one leg to GPIO33, the other leg to GND. No resistor needed (internal pull-up) |

```
ESP32 (the pins on the left/right edge are labelled D25, D26, D27, D32, D33, D21, D22, 3V3, GND)

   GPIO25 --[330]--(>|)-- GND        green LED
   GPIO26 --[330]--(>|)-- GND        amber LED
   GPIO27 --[330]--(>|)-- GND        red LED
   GPIO32 ---------- buzzer I/O
   GPIO33 ---------- button ---------- GND
   GPIO21 -- OLED SDA      GPIO22 -- OLED SCL      3V3 -- OLED VCC      GND -- OLED GND
```
Use the breadboard's blue (-) rail as the common ground: all LED short legs, the buzzer GND, the button and the OLED GND go to it, and the rail goes to one ESP32 **GND** pin.

## Buzzer
- **Buzzer module with 3 pins (VCC, I/O, GND)**: VCC -> 3V3, I/O -> GPIO32, GND -> GND. Easiest.
  If it beeps all the time when idle, it is an "active-low" module: set `BUZZER_ACTIVE_LOW true` in the sketch.
- **Bare 2-pin ACTIVE buzzer** (makes a tone by itself when powered): drive it with a transistor so the pin is not overloaded:
  GPIO32 -> 1 kohm -> base of an NPN transistor (BC547 / 2N2222); emitter -> GND; collector -> buzzer (-); buzzer (+) -> **VIN/5V** pin of the ESP32 (when powered over USB).
- **PASSIVE buzzer** (needs a tone, stays silent on plain DC): same wiring as above, and set `PASSIVE_BUZZER true` in the sketch.
  Not sure which you have? Touch the two legs to a 3V battery: a continuous beep means active; only a click means passive.

## Start-up check
When the ESP32 powers up, the green, amber and red LEDs flash once in turn and the buzzer beeps once. If one LED does not flash, check its direction (long leg towards the resistor/GPIO) and its pin.
If the OLED stays dark, change `OLED_ADDR` in the sketch from `0x3C` to `0x3D`. The LEDs and buzzer keep working even without the OLED.
