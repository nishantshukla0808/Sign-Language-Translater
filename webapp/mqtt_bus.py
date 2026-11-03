"""
MQTT publisher for the SignTalk web server (fails soft: if the broker is down the app keeps working).

Topics (all under "signtalk/"):
  status      retained   "online" / "offline"   (offline is the broker's last-will if the server dies)
  phrase      QoS 1      {"id","label","p","ms","ts"}          an accepted, spoken phrase
  unsure      QoS 0      {"reason","label","p","ts"}           a sign that was not accepted
  telemetry   retained   {"temp","load","uptime","predictions","spoken","phrases","ts"}  every 5 s
  alert       retained   {"state":"on"|"off","source":...}      emergency button (published by iot/gpio_bridge.py)
"""

from __future__ import annotations

import json
import os
import threading
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:                                  # paho-mqtt not installed: MQTT is simply off
    mqtt = None

T = "signtalk/"


def _dumps(o) -> str:
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


class Bus:
    def __init__(self, host: str, port: int, user: str, password: str, telemetry=None, client_id: str = "signtalk-server", prefix: str = T):
        self.t = prefix
        self.enabled = mqtt is not None and bool(host)
        self.connected = False
        self.predictions = 0
        self.spoken = 0
        self.t0 = time.time()
        self._telemetry = telemetry                  # callable -> dict with extra fields (temp, load, phrases)
        if not self.enabled:
            print("[mqtt] disabled" + (" (paho-mqtt is not installed)" if mqtt is None else ""), flush=True)
            return
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.c.username_pw_set(user, password)
        self.c.will_set(self.t + "status", "offline", qos=1, retain=True)
        self.c.on_connect = self._on_connect
        self.c.on_disconnect = self._on_disconnect
        self.c.reconnect_delay_set(1, 15)
        self.c.connect_async(host, port, keepalive=30)
        self.c.loop_start()
        threading.Thread(target=self._telemetry_loop, daemon=True, name="mqtt-telemetry").start()

    # --- connection callbacks
    def _on_connect(self, c, userdata, flags, rc, props=None):
        if rc == 0 or getattr(rc, "value", 1) == 0:
            self.connected = True
            c.publish(self.t + "status", "online", qos=1, retain=True)
            print("[mqtt] connected to the broker", flush=True)
        else:
            print(f"[mqtt] broker refused the connection: {rc}", flush=True)

    def _on_disconnect(self, c, userdata, flags, rc, props=None):
        if self.connected:
            print("[mqtt] disconnected, retrying in the background", flush=True)
        self.connected = False

    # --- publishing (never raises)
    def _pub(self, topic: str, payload, qos: int = 0, retain: bool = False):
        if not self.enabled:
            return
        try:
            self.c.publish(self.t + topic, payload if isinstance(payload, str) else _dumps(payload), qos=qos, retain=retain)
        except Exception as e:                      # noqa: BLE001
            print(f"[mqtt] publish failed: {e}", flush=True)

    def prediction(self, res: dict):
        """Called with the result dict of /api/predict."""
        if res.get("reason") in ("too_short", "no_body", "no_hands") and "best" not in res:
            return                                   # nothing was classified; not worth an event
        self.predictions += 1
        best = res.get("best") or {}
        ts = int(time.time())
        if res.get("accepted"):
            self.spoken += 1
            self._pub("phrase", {"id": best.get("id"), "label": best.get("label"), "p": best.get("p"), "ms": res.get("ms"), "ts": ts}, qos=1)
        else:
            self._pub("unsure", {"reason": res.get("reason", "unsure"), "label": best.get("label"), "p": best.get("p"), "ts": ts})

    def _telemetry_loop(self):
        while True:
            time.sleep(5)
            if not self.connected:
                continue
            d = {"uptime": int(time.time() - self.t0), "predictions": self.predictions, "spoken": self.spoken, "ts": int(time.time())}
            try:
                if self._telemetry:
                    d.update(self._telemetry())
            except Exception:                        # noqa: BLE001
                pass
            self._pub("telemetry", d, retain=True)

    def close(self):
        if self.enabled:
            try:
                self.c.publish(self.t + "status", "offline", qos=1, retain=True).wait_for_publish(2)
                self.c.loop_stop(); self.c.disconnect()
            except Exception:                        # noqa: BLE001
                pass
