# SignTalk IoT layer: MQTT + ESP32 (OLED, LEDs, buzzer, button)

```
 phone page --> web server (webapp/server.py) --publishes--> MQTT broker (Mosquitto, port 1883, on the Pi)
                                                               '-- ESP32 (iot/esp32_oled/esp32_oled.ino): 0.96" OLED shows the phrase,
                                                                   green / amber / red LEDs, buzzer, emergency button (button publishes signtalk/alert)
```

## Topics (login: user `signtalk`, password `12345678`)
| Topic | Retained | Payload | Who publishes |
|---|---|---|---|
| `signtalk/phrase` | no (QoS 1) | `{"id":"hello","label":"Hello","p":0.93,"ms":2.4,"ts":...}` | web server, when a phrase is accepted and spoken |
| `signtalk/unsure` | no | `{"reason":"unsure","label":"Hi","p":0.41,"ts":...}` | web server, when a sign was seen but not accepted |
| `signtalk/status` | yes | `online` / `offline` | web server (`offline` is its MQTT last-will) |
| `signtalk/telemetry` | yes | `{"temp":49.0,"load":0.4,"uptime":..,"predictions":..,"spoken":..,"phrases":8}` every 5 s | web server |
| `signtalk/alert` | yes | `{"state":"on"\|"off","source":"button"}` | the ESP32 (button) - or anyone, e.g. `mosquitto_pub` |

## Run it (on the Pi)
```bash
cd ~/nishu_hand_skeleton/signtalk
iot/broker.sh start                       # 1. MQTT broker   (iot/broker.sh status | stop | test)
.venv/bin/python webapp/server.py         # 2. web server (connects to the broker by itself)
```
The Pi needs nothing else. Power the ESP32 (USB) and it connects by itself. At power-up it flashes each LED once and beeps once: that is the wiring check.
(`iot/gpio_bridge.py` is an older option that drives LEDs/buzzer/button from the Pi's own GPIO instead. It is not used now.)

What you should see: **green LED + short beep** when a phrase is recognised, **amber LED** when a sign is not accepted, **red LED blinking slowly** while the web server is offline, **press the button** = emergency alert (red LED fast blink + alarm beeps, the OLED flashes "!! HELP !!"; all on the ESP32), press again to clear it. An alert also clears itself after 60 s.

Watch the messages live: `LD_LIBRARY_PATH=iot/broker/usr/lib/aarch64-linux-gnu iot/broker/usr/bin/mosquitto_sub -h 127.0.0.1 -u signtalk -P 12345678 -t 'signtalk/#' -v`

## ESP32 (OLED + LEDs + buzzer + button)
1. Wire everything as in `iot/WIRING.md` (OLED SDA GPIO21 / SCL GPIO22; LEDs GPIO25/26/27; buzzer GPIO32; button GPIO33 to GND).
2. Arduino IDE: add the ESP32 board package, install the libraries *PubSubClient*, *Adafruit SSD1306* and *Adafruit GFX Library*.
3. Open `iot/esp32_oled/esp32_oled.ino`, set `WIFI_SSID`, `WIFI_PASS` and `MQTT_HOST` (the Pi's IP; the ESP32 needs a **2.4 GHz** network) at the top, choose board *ESP32 Dev Module*, upload. Buzzer type: `PASSIVE_BUZZER` / `BUZZER_ACTIVE_LOW` just below the pin numbers.
4. The screen shows Wi-Fi / MQTT status and the Pi's temperature, "Waiting for sign", then the phrase in big letters with a confidence bar for 8 s.

## Broker without sudo
`broker.sh` uses a system Mosquitto if there is one (`sudo apt install mosquitto mosquitto-clients`). Otherwise it uses the copy unpacked in `iot/broker/`, which was fetched without sudo:
```bash
mkdir -p /tmp/mq && cd /tmp/mq && apt-get download mosquitto mosquitto-clients libmosquitto1 libdlt2 libwebsockets19t64
for f in *.deb; do dpkg -x $f ~/nishu_hand_skeleton/signtalk/iot/broker; done
```
Security note: the broker listens on all interfaces with a simple password; it is meant for a private Wi-Fi (home router or the Pi's own password-protected hotspot). Do not expose port 1883 to the internet.

## Tests (no hardware needed)
```bash
.venv/bin/python iot/test_iot.py     # MQTT publisher, retained messages, wrong password (and the older Pi-GPIO bridge with simulated pins)
```
