"""
Tests for the MQTT + GPIO part, no hardware needed (GPIO is simulated with gpiozero's mock pins).
Needs the broker running:  iot/broker.sh start      Run:  .venv/bin/python iot/test_iot.py
It uses its own topic prefix (signtalk-test/), so it is safe to run while the real system is up.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ["GPIOZERO_PIN_FACTORY"] = "mock"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paho.mqtt.client as mqtt                       # noqa: E402

import gpio_bridge                                     # noqa: E402
from mqtt_bus import Bus                               # noqa: E402

HOST, PORT, USER, PW = "127.0.0.1", 1883, "signtalk", os.environ.get("MQTT_PASS", "12345678")
PFX = "signtalk-test/"          # separate namespace: never touches the live system or its LEDs
fails = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        fails.append(name)


def client(cid):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
    c.username_pw_set(USER, PW)
    c.connect(HOST, PORT, 10)
    c.loop_start()
    return c


def sample(dev, secs, step=0.05):
    seen = set(); t = time.time()
    while time.time() - t < secs:
        seen.add(int(dev.value)); time.sleep(step)
    return seen


def wait(cond, secs=3.0):
    t = time.time()
    while time.time() - t < secs:
        if cond():
            return True
        time.sleep(0.05)
    return False


print("broker")
bad = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="bad")
rc = []
bad.on_connect = lambda c, u, f, r, p=None: rc.append(r)
bad.username_pw_set(USER, "wrong-password"); bad.connect(HOST, PORT, 5); bad.loop_start(); time.sleep(1); bad.loop_stop()
check("wrong password is refused", rc and getattr(rc[0], "value", rc[0]) != 0)

print("server publisher (webapp/mqtt_bus.py)")
clr = client("test-clear")                                  # leftovers from an earlier run would break the exact-match checks
for t in ("status", "telemetry", "alert"):
    clr.publish(PFX + t, b"", qos=1, retain=True).wait_for_publish()
clr.loop_stop(); clr.disconnect()
got = {}
sub = client("test-sub")
sub.on_message = lambda c, u, m: got.setdefault(m.topic, []).append(m.payload.decode())
sub.subscribe(PFX + "#", 1)
bus = Bus(HOST, PORT, USER, PW, telemetry=lambda: {"temp": 50.0, "load": 0.1, "phrases": 8}, client_id="test-bus", prefix=PFX)
check("status online (retained)", wait(lambda: got.get(PFX + "status") == ["online"]))
bus.prediction({"accepted": True, "best": {"id": "hello", "label": "Hello", "p": 0.93}, "ms": 2.4})
bus.prediction({"accepted": False, "reason": "unsure", "best": {"id": "hii", "label": "Hi", "p": 0.4}})
bus.prediction({"accepted": False, "reason": "no_hands"})            # nothing classified: must not be published
check("accepted phrase published", wait(lambda: PFX + "phrase" in got))
ph = json.loads(got[PFX + "phrase"][0]); check("phrase payload", ph["id"] == "hello" and ph["label"] == "Hello" and ph["p"] == 0.93, str(ph))
check("unsure published, empty windows are not", wait(lambda: PFX + "unsure" in got) and len(got[PFX + "unsure"]) == 1)
check("counters", (bus.predictions, bus.spoken) == (2, 1))
bus._pub("telemetry", {"uptime": 1}, retain=True)
check("telemetry reaches subscribers", wait(lambda: PFX + "telemetry" in got))
late = []
l = client("late"); l.on_message = lambda c, u, m: late.append((m.topic, m.payload.decode())); l.subscribe(PFX + "status", 1)
check("a late subscriber gets the retained status", wait(lambda: (PFX + "status", "online") in late)); l.loop_stop()

print("GPIO bridge (mock pins)")
out = gpio_bridge.Outputs()
br = gpio_bridge.Bridge(out, HOST, PORT, USER, PW, alert_timeout=0, prefix=PFX)
time.sleep(1)
pub = client("test-pub")
pub.publish(PFX + "alert", json.dumps({"state": "off"}), qos=1, retain=True); time.sleep(0.3)
pub.publish(PFX + "phrase", json.dumps({"id": "hello", "label": "Hello", "p": 0.9}), qos=1)
check("phrase -> green LED on", wait(lambda: out.green.value == 1))
check("phrase -> buzzer beeped", wait(lambda: any(s.state for s in out.buzzer.pin.states)))
check("green LED turns off again", wait(lambda: out.green.value == 0, 4.0))
pub.publish(PFX + "unsure", json.dumps({"reason": "unsure"}))
check("unsure -> amber LED on", wait(lambda: out.amber.value == 1))
check("amber LED turns off again", wait(lambda: out.amber.value == 0, 3.0))
pub.publish(PFX + "status", "offline", qos=1, retain=True)
check("server offline -> red LED blinks", sample(out.red, 2.5) == {0, 1})
pub.publish(PFX + "status", "online", qos=1, retain=True); time.sleep(1.2)
check("server online -> red LED off", sample(out.red, 1.0) == {0})
n0 = len(out.buzzer.pin.states)
pub.publish(PFX + "alert", json.dumps({"state": "on", "source": "test"}), qos=1)
check("alert on -> red LED + buzzer alarm", sample(out.red, 1.5) == {0, 1} and len(out.buzzer.pin.states) > n0 + 2)
pub.publish(PFX + "alert", json.dumps({"state": "off"}), qos=1, retain=True)
check("alert off -> quiet again", wait(lambda: sample(out.red, 0.6) == {0}, 3.0))

print("emergency button")
got.pop(PFX + "alert", None)
out.button.pin.drive_low(); time.sleep(0.2); out.button.pin.drive_high()      # press and release
check("button press -> alert ON published (retained)", wait(lambda: got.get(PFX + "alert") and json.loads(got[PFX + "alert"][-1])["state"] == "on"))
check("button press -> red LED blinking", sample(out.red, 1.2) == {0, 1})
time.sleep(0.3); out.button.pin.drive_low(); time.sleep(0.2); out.button.pin.drive_high()
check("second press -> alert OFF", wait(lambda: json.loads(got[PFX + "alert"][-1])["state"] == "off"))
check("second press -> quiet again", wait(lambda: sample(out.red, 0.6) == {0}, 3.0))

for tp in ("status", "alert", "telemetry"):                                    # clear retained test state
    pub.publish(PFX + tp, "", qos=1, retain=True)
time.sleep(0.3)
bus.close(); br.close(); out.close(); sub.loop_stop(); pub.loop_stop()
print("\nFAILED: " + ", ".join(fails) if fails else "\nALL PASSED")
sys.exit(1 if fails else 0)
