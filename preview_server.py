"""
Live camera preview in a browser, for a Pi with no display.

    .venv/bin/python preview_server.py [--camera 0] [--port 8081] [--rotate 0]

Open http://<pi-ip>:<port>/ on any device on the same network. Stop it (Ctrl+C)
before running realtime.py: only one program can hold the camera.
A thin box marks the area to keep your head, shoulders and hands inside.
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

p = argparse.ArgumentParser()
p.add_argument("--camera", default="0", help="index or stream URL")
p.add_argument("--port", type=int, default=8081)
p.add_argument("--width", type=int, default=640)
p.add_argument("--height", type=int, default=480)
p.add_argument("--mjpg", action="store_true", help="request MJPG (needed for 1280x720+ on most webcams)")
p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
args = p.parse_args()

src = int(args.camera) if args.camera.isdigit() else args.camera
cap = cv2.VideoCapture(src)
if isinstance(src, int):
    if args.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
latest = {"jpg": None}


def grab():
    while True:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        if args.rotate:
            f = ROT[args.rotate](f)
        h, w = f.shape[:2]
        cv2.rectangle(f, (int(w * 0.1), int(h * 0.05)), (int(w * 0.9), int(h * 0.95)), (0, 255, 0), 1)
        latest["jpg"] = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = latest["jpg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j))
                        self.wfile.write(j + b"\r\n")
                    time.sleep(0.07)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='max-width:100%'></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


threading.Thread(target=grab, daemon=True).start()
print(f"preview on http://0.0.0.0:{args.port}/   (Ctrl+C to stop)", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
