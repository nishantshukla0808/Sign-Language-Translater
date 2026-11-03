/*
  SignTalk ESP32 node: 0.96" OLED + 3 LEDs + buzzer + emergency button, all on the ESP32.
  Subscribes to the Raspberry Pi's MQTT broker (the Pi only runs the model and the broker):
    phrase recognised -> green LED + short beep + phrase on the OLED
    sign not accepted -> amber LED
    Pi server offline / Wi-Fi or broker lost -> red LED blinks slowly
    emergency button  -> publishes signtalk/alert; red LED fast blink + alarm beeps + OLED "!! HELP !!" (toggle; clears itself after 60 s)

  Libraries (Arduino IDE > Library Manager):  PubSubClient (Nick O'Leary), Adafruit SSD1306, Adafruit GFX
  Board: "ESP32 Dev Module"   (ESP32 Wi-Fi is 2.4 GHz only)

  OLED (SSD1306, I2C, 128x64):  VCC -> 3V3,  GND -> GND,  SDA -> GPIO21,  SCL -> GPIO22
  LEDs (each through a 220-330 ohm resistor to GND):  green GPIO25, amber GPIO26, red GPIO27
  Buzzer: GPIO32 (see iot/WIRING.md)      Button: GPIO33 to GND (internal pull-up, no resistor)
  Topics: signtalk/phrase  signtalk/unsure  signtalk/status  signtalk/telemetry  signtalk/alert
*/
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ======== EDIT THESE ========
const char* WIFI_SSID = "YOUR_WIFI_NAME";        // the same network as the Pi (2.4 GHz). Hotspot mode: "SignTalk"
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";    // hotspot mode: "12345678"
const char* MQTT_HOST = "192.168.29.66";         // the Pi's IP address (hotspot mode: 10.42.0.1)
const uint16_t MQTT_PORT = 1883;
const char* MQTT_USER = "signtalk";
const char* MQTT_PASS = "12345678";
// ============================

#define SDA_PIN 21
#define SCL_PIN 22
#define OLED_ADDR 0x3C            // some boards use 0x3D
#define PHRASE_SHOW_MS 8000       // how long a recognised phrase stays big on the screen

#define GREEN_PIN 25
#define AMBER_PIN 26
#define RED_PIN 27
#define BUZZER_PIN 32
#define BUTTON_PIN 33
#define PASSIVE_BUZZER false      // true if the buzzer only sounds when given a tone
#define BUZZER_ACTIVE_LOW false   // true for a 3-pin module that beeps all the time with the default
#define ALERT_TIMEOUT_MS 60000    // an alert switches itself off after this long

Adafruit_SSD1306 oled(128, 64, &Wire, -1);
WiFiClient net;
PubSubClient mqtt(net);

String lastLabel = "";            // most recent recognised phrase
int lastPct = 0;
unsigned long lastPhraseAt = 0;
unsigned long unsureAt = 0;
String piTemp = "", piStatus = "?";
bool alertOn = false;
bool oledOk = false;
unsigned long lastMqttTry = 0, lastDraw = 0;
unsigned long greenUntil = 0, amberUntil = 0, beepUntil = 0, alertSince = 0;
bool lastButton = HIGH;
unsigned long buttonChangedAt = 0;

// ---- buzzer / LEDs
void buzzer(bool on) {
#if PASSIVE_BUZZER
  #if ESP_ARDUINO_VERSION_MAJOR >= 3
    if (on) tone(BUZZER_PIN, 2000); else noTone(BUZZER_PIN);
  #else
    if (on) ledcWriteTone(0, 2000); else ledcWriteTone(0, 0);
  #endif
#else
  digitalWrite(BUZZER_PIN, (on != BUZZER_ACTIVE_LOW) ? HIGH : LOW);
#endif
}

void setupOutputs() {
  pinMode(GREEN_PIN, OUTPUT); pinMode(AMBER_PIN, OUTPUT); pinMode(RED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
#if PASSIVE_BUZZER && ESP_ARDUINO_VERSION_MAJOR < 3
  ledcSetup(0, 2000, 8); ledcAttachPin(BUZZER_PIN, 0);
#else
  pinMode(BUZZER_PIN, OUTPUT);
#endif
  buzzer(false);
}

// flash every LED once and beep once at start-up: a quick wiring check
void selfTest() {
  const int leds[3] = {GREEN_PIN, AMBER_PIN, RED_PIN};
  for (int i = 0; i < 3; i++) { digitalWrite(leds[i], HIGH); delay(250); digitalWrite(leds[i], LOW); }
  buzzer(true); delay(150); buzzer(false);
}

void setAlert(bool on) {
  if (on && !alertOn) alertSince = millis();
  alertOn = on;
}

void publishAlert(bool on) {
  if (!mqtt.connected()) return;
  String p = String("{\"state\":\"") + (on ? "on" : "off") + "\",\"source\":\"button\"}";
  mqtt.publish("signtalk/alert", p.c_str(), true);        // retained, so a restarted node learns the state
}

// button: press = toggle the emergency alert (debounced, on the falling edge)
void checkButton() {
  bool b = digitalRead(BUTTON_PIN);
  if (b != lastButton && millis() - buttonChangedAt > 50) {
    buttonChangedAt = millis();
    lastButton = b;
    if (b == LOW) { setAlert(!alertOn); publishAlert(alertOn); }
  }
}

// drive LEDs and buzzer from the current state (called every loop, never blocks)
void updateOutputs() {
  unsigned long now = millis();
  if (alertOn && now - alertSince > ALERT_TIMEOUT_MS) { setAlert(false); publishAlert(false); }
  bool offline = (WiFi.status() != WL_CONNECTED) || !mqtt.connected() || piStatus == "offline";
  digitalWrite(GREEN_PIN, now < greenUntil);
  digitalWrite(AMBER_PIN, now < amberUntil);
  if (alertOn) {                                          // fast blink + alarm beeps, 2 per second
    bool on = (now / 250) % 2 == 0;
    digitalWrite(RED_PIN, on);
    buzzer(on);
  } else {
    digitalWrite(RED_PIN, offline && (now / 1000) % 2 == 0);   // slow blink while the server is not reachable
    buzzer(now < beepUntil);
  }
}

// minimal JSON value reader: finds "key": value in a flat JSON object (string or number)
String jget(const String& s, const char* key) {
  String k = String("\"") + key + "\":";
  int i = s.indexOf(k);
  if (i < 0) return "";
  i += k.length();
  while (i < (int)s.length() && s[i] == ' ') i++;
  if (i < (int)s.length() && s[i] == '"') {
    int j = s.indexOf('"', i + 1);
    return j < 0 ? "" : s.substring(i + 1, j);
  }
  int j = i;
  while (j < (int)s.length() && s[j] != ',' && s[j] != '}') j++;
  return s.substring(i, j);
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  String t(topic), p;
  for (unsigned int i = 0; i < len; i++) p += (char)payload[i];
  if (t == "signtalk/phrase") {
    lastLabel = jget(p, "label");
    lastPct = (int)(jget(p, "p").toFloat() * 100.0f + 0.5f);
    lastPhraseAt = millis();
    greenUntil = lastPhraseAt + 2000;
    beepUntil = lastPhraseAt + 120;
  } else if (t == "signtalk/unsure") {
    unsureAt = millis();
    amberUntil = unsureAt + 1500;
  } else if (t == "signtalk/status") {
    piStatus = p;
  } else if (t == "signtalk/telemetry") {
    piTemp = jget(p, "temp");
  } else if (t == "signtalk/alert") {
    setAlert(jget(p, "state") == "on");
  }
}

unsigned long lastWifiBegin = 0;

// tells you WHY the connection fails (see the number in the Serial Monitor)
void onWifiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    int r = info.wifi_sta_disconnected.reason;
    Serial.print("Wi-Fi: not connected, reason "); Serial.print(r);
    if (r == 201) Serial.println("  (network name not found: wrong SSID, out of range, or 5 GHz only)");
    else if (r == 202 || r == 15 || r == 204 || r == 2) Serial.println("  (wrong password, or the router refuses this security mode)");
    else Serial.println();
  } else if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    Serial.print("Wi-Fi: connected, ESP32 address "); Serial.println(WiFi.localIP());
  }
}

void startWifi() {
  WiFi.persistent(false);                    // do not wear the flash with saved credentials
  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                      // modem sleep can make joining and MQTT flaky
  WiFi.setAutoReconnect(true);               // the ESP32 retries by itself after a drop
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  lastWifiBegin = millis();
  Serial.print("Wi-Fi: connecting to "); Serial.println(WIFI_SSID);
}

void connectWifi() {                         // safety net only: restart the attempt if nothing worked for 30 s
  if (WiFi.status() == WL_CONNECTED) { lastWifiBegin = millis(); return; }
  if (millis() - lastWifiBegin < 30000) return;
  lastWifiBegin = millis();
  WiFi.disconnect(false, false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.println("Wi-Fi: still not connected, trying again");
}

void connectMqtt() {                         // non-blocking: one attempt every 3 s
  if (WiFi.status() != WL_CONNECTED || mqtt.connected() || millis() - lastMqttTry < 3000) return;
  lastMqttTry = millis();
  String id = "signtalk-esp32-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  if (mqtt.connect(id.c_str(), MQTT_USER, MQTT_PASS)) {
    mqtt.subscribe("signtalk/phrase", 1);
    mqtt.subscribe("signtalk/unsure");
    mqtt.subscribe("signtalk/status", 1);
    mqtt.subscribe("signtalk/telemetry");
    mqtt.subscribe("signtalk/alert", 1);
  }
}

// draw text centred on the display at the given y
void centred(const String& s, int y, int size) {
  oled.setTextSize(size);
  int16_t x1, y1; uint16_t w, h;
  oled.getTextBounds(s, 0, 0, &x1, &y1, &w, &h);
  oled.setCursor((128 - (int)w) / 2, y);
  oled.print(s);
}

// show a phrase as big as it fits: one line at size 2 (10 chars), two lines split at a space, else size 1
void bigPhrase(const String& s, int top) {
  if (s.length() <= 10) { centred(s, top + 8, 2); return; }
  int sp = s.indexOf(' ');
  if (sp > 0 && sp <= 10 && (int)s.length() - sp - 1 <= 10) {
    centred(s.substring(0, sp), top, 2);
    centred(s.substring(sp + 1), top + 18, 2);
  } else {
    centred(s, top + 12, 1);
  }
}

void draw() {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  unsigned long now = millis();

  if (alertOn) {                             // emergency: flashing inverted screen
    bool inv = (now / 400) % 2 == 0;
    oled.fillRect(0, 0, 128, 64, inv ? SSD1306_WHITE : SSD1306_BLACK);
    oled.setTextColor(inv ? SSD1306_BLACK : SSD1306_WHITE);
    centred("!! HELP !!", 10, 2);
    centred("emergency button", 42, 1);
    oled.display();
    return;
  }

  // status bar
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.print(WiFi.status() == WL_CONNECTED ? "WiFi" : "WiFi?");
  oled.print(mqtt.connected() ? " MQTT" : " MQTT?");
  if (piTemp.length()) { oled.setCursor(92, 0); oled.print(piTemp.substring(0, 4)); oled.print("C"); }
  oled.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (WiFi.status() != WL_CONNECTED) {
    centred("Connecting Wi-Fi", 28, 1);
  } else if (!mqtt.connected()) {
    centred("Connecting to Pi", 28, 1);
  } else if (piStatus == "offline") {
    centred("Pi server offline", 28, 1);
  } else if (lastLabel.length() && now - lastPhraseAt < PHRASE_SHOW_MS) {
    bigPhrase(lastLabel, 16);
    int w = constrain(lastPct, 0, 100);                    // confidence bar, 100 px wide
    oled.drawRect(4, 52, 102, 8, SSD1306_WHITE);
    oled.fillRect(5, 53, w, 6, SSD1306_WHITE);
    oled.setCursor(108, 52); oled.print(lastPct); oled.print("%");
  } else {
    centred(unsureAt && now - unsureAt < 2500 ? "Not sure..." : "Waiting for sign", 22, 1);
    if (lastLabel.length()) {
      oled.setCursor(0, 44); oled.print("Last:");
      oled.setCursor(0, 54); oled.print(lastLabel.substring(0, 21));
    }
  }
  oled.display();
}

void setup() {
  Serial.begin(115200);
  setupOutputs();
  selfTest();
  Wire.begin(SDA_PIN, SCL_PIN);
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (oledOk) {
    oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE);
    centred("SignTalk", 14, 2); centred("ISL to speech", 40, 1);
    oled.display();
    delay(1200);
  } else {
    Serial.println("OLED not found - check wiring and the I2C address (LEDs and buzzer still work)");
  }
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setBufferSize(512);
  mqtt.setCallback(onMessage);
  startWifi();
}

void loop() {
  connectWifi();                              // retries by itself if the router restarts
  connectMqtt();
  mqtt.loop();
  checkButton();
  updateOutputs();
  if (oledOk && millis() - lastDraw > 50) { lastDraw = millis(); draw(); }
}
