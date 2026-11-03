"""
GPIO bridge: MQTT <-> LEDs, buzzer and emergency button on the Raspberry Pi header.

    .venv/bin/python iot/gpio_bridge.py [--no-selftest] [--passive-buzzer]

Subscribes (broker on this Pi):
    signtalk/phrase   -> green LED for 2 s + one short beep           (a phrase was recognised and spoken)
    signtalk/unsure   -> amber LED for 1.5 s                         (a sign was seen but not accepted)
    signtalk/status   -> red LED blinks slowly while the web server is "offline"
    signtalk/alert    -> state "on": red LED fast blink + alarm beeps until state "off"
Publishes:
    signtalk/alert    {"state":"on"|"off","source":"button"}   (retained) when the button is pressed

Wiring (BCM numbers, see iot/WIRING.md):  green 17, amber 27, red 22, buzzer 18, button 23 (to GND).
Run it on the Pi only. With GPIOZERO_PIN_FACTORY=mock it runs without hardware (used by the tests).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import paho.mqtt.client as mqtt
from gpiozero import LED, Button, Buzzer, Device, PWMOutputDevice

T = "signtalk/"


class Outputs:
    """The five GPIO devices and the patterns on them. All patterns run on one worker thread so they never overlap badly."""

    def __init__(self, green=17, amber=27, red=22, buzzer=18, button=23, passive=False, pwm_hz=2000, buzzer_active_high=True):
        self.green, self.amber, self.red = LED(green), LED(amber), LED(red)
        self.passive = passive
        self.buzzer = (PWMOutputDevice(buzzer, frequency=pwm_hz, initial_value=0, active_high=buzzer_active_high) if passive
                       else Buzzer(buzzer, active_high=buzzer_active_high))
        self.button = Button(button, pull_up=True, bounce_time=0.05)
        self.alert = False
        self.server_online = True
        self._lock = threading.Lock()
        self._green_until = self._amber_until = 0.0
        self._stop = threading.Event()
        self._beeps: list[tuple[float, float]] = []          # (on seconds, off seconds)
        self._worker = threading.Thread(target=self._run, daemon=True, name="gpio-worker")
        self._worker.start()

    # --- requests (thread-safe, return immediately)
    def recognised(self, hold: float = 2.0):
        with self._lock:
            self._green_until = time.monotonic() + hold
            self._beeps += [(0.12, 0.0)]

    def unsure(self, hold: float = 1.5):
        with self._lock:
            self._amber_until = time.monotonic() + hold

    def set_alert(self, on: bool):
        with self._lock:
            self.alert = on

    def set_server_online(self, online: bool):
        with self._lock:
            self.server_online = online

    # --- device helpers
    def _buz(self, on: bool):
        if self.passive:
            self.buzzer.value = 0.5 if on else 0.0
        elif on:
            self.buzzer.on()
        else:
            self.buzzer.off()

    def selftest(self):
        """Light every LED in turn and beep once, so wiring mistakes are obvious at start-up."""
        for led in (self.green, self.amber, self.red):
            led.on(); time.sleep(0.25); led.off()
        self._buz(True); time.sleep(0.15); self._buz(False)

    def _run(self):
        tick = 0
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                alert, online = self.alert, self.server_online
                g, a = now < self._green_until, now < self._amber_until
                beep = self._beeps.pop(0) if self._beeps else None
            self.green.value = 1 if g else 0
            self.amber.value = 1 if a else 0
            if alert:                                              # fast blink + alarm beeps (2 per second)
                on = (tick // 2) % 2 == 0
                self.red.value = 1 if on else 0
                self._buz(on)
            else:
                self.red.value = 1 if (not online and (tick // 10) % 2 == 0) else 0   # slow blink when the server is offline
                if beep:
                    self._buz(True); time.sleep(beep[0]); self._buz(False)
            tick += 1
            time.sleep(0.1)

    def close(self):
        self._stop.set()
        self._worker.join(1)
        for d in (self.green, self.amber, self.red, self.buzzer, self.button):
            try:
                d.close()
            except Exception:                                      # noqa: BLE001
                pass


class Bridge:
    def __init__(self, out: Outputs, host: str, port: int, user: str, password: str, alert_timeout: float = 60.0, prefix: str = T):
        self.out, self.alert_timeout, self.t = out, alert_timeout, prefix
        self._alert_timer: threading.Timer | None = None
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="signtalk-gpio-bridge")
        self.c.username_pw_set(user, password)
        self.c.on_connect = self._on_connect
        self.c.on_disconnect = lambda *a: self.out.set_server_online(False)
        self.c.on_message = self._on_message
        self.c.reconnect_delay_set(1, 15)
        self.c.connect_async(host, port, keepalive=30)
        out.button.when_pressed = self._button
        self.c.loop_start()

    def _on_connect(self, c, userdata, flags, rc, props=None):
        c.subscribe([(self.t + "phrase", 1), (self.t + "unsure", 0), (self.t + "status", 1), (self.t + "alert", 1)])
        print("[bridge] connected to the broker, listening", flush=True)

    def _on_message(self, c, userdata, msg):
        payload = msg.payload.decode("utf-8", "replace")
        topic = msg.topic[len(self.t):]
        if topic == "phrase":
            self.out.recognised()
            try:
                print(f"[bridge] phrase: {json.loads(payload).get('label')}", flush=True)
            except ValueError:
                pass
        elif topic == "unsure":
            self.out.unsure()
        elif topic == "status":
            self.out.set_server_online(payload.strip() == "online")
        elif topic == "alert":
            try:
                self._apply_alert(json.loads(payload).get("state") == "on")
            except ValueError:
                pass

    def _apply_alert(self, on: bool):
        self.out.set_alert(on)
        if self._alert_timer:
            self._alert_timer.cancel()
            self._alert_timer = None
        if on and self.alert_timeout:
            self._alert_timer = threading.Timer(self.alert_timeout, lambda: self.set_alert(False))
            self._alert_timer.daemon = True
            self._alert_timer.start()

    def set_alert(self, on: bool, publish: bool = True):
        self._apply_alert(on)
        if publish:
            self.c.publish(self.t + "alert", json.dumps({"state": "on" if on else "off", "source": "button", "ts": int(time.time())},
                                                   separators=(",", ":")), qos=1, retain=True)
        print(f"[bridge] ALERT {'ON' if on else 'off'}", flush=True)

    def _button(self):
        self.set_alert(not self.out.alert)             # press = toggle the emergency alert

    def close(self):
        self.c.loop_stop()
        self.c.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("MQTT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    ap.add_argument("--user", default=os.environ.get("MQTT_USER", "signtalk"))
    ap.add_argument("--password", default=os.environ.get("MQTT_PASS", "12345678"))
    for n, d in (("green", 17), ("amber", 27), ("red", 22), ("buzzer", 18), ("button", 23)):
        ap.add_argument(f"--{n}-pin", type=int, default=d, help=f"BCM pin of the {n} (default {d})")
    ap.add_argument("--passive-buzzer", action="store_true", help="buzzer needs a tone (passive type); default is an active buzzer")
    ap.add_argument("--buzzer-active-low", action="store_true", help="buzzer module that sounds when the pin is LOW (it beeps all the time with the default)")
    ap.add_argument("--alert-timeout", type=float, default=60.0, help="seconds until an alert switches itself off (0 = never)")
    ap.add_argument("--no-selftest", action="store_true")
    a = ap.parse_args()
    if os.environ.get("GPIOZERO_PIN_FACTORY") != "mock":
        from gpiozero.pins.lgpio import LGPIOFactory          # Raspberry Pi 5 needs lgpio, not RPi.GPIO
        Device.pin_factory = LGPIOFactory(chip=0)
    out = Outputs(a.green_pin, a.amber_pin, a.red_pin, a.buzzer_pin, a.button_pin, passive=a.passive_buzzer,
                  buzzer_active_high=not a.buzzer_active_low)
    if not a.no_selftest:
        out.selftest()
    br = Bridge(out, a.host, a.port, a.user, a.password, a.alert_timeout)
    print(f"[bridge] GPIO: green {a.green_pin}, amber {a.amber_pin}, red {a.red_pin}, buzzer {a.buzzer_pin}, button {a.button_pin} (BCM)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        br.close(); out.close()


if __name__ == "__main__":
    main()
