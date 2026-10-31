"""Tiny MJPEG server: lets you watch the annotated camera feed in a browser (Pi with no display)."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class StreamServer:
    def __init__(self, port: int):
        self.jpg: bytes | None = None
        outer = self

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
                            j = outer.jpg
                            if j:
                                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j))
                                self.wfile.write(j + b"\r\n")
                            time.sleep(0.07)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    body = (b"<html><body style='margin:0;background:#111'>"
                            b"<img src='/stream' style='width:100%;max-width:1100px'></body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="stream").start()

    def publish(self, bgr) -> None:
        self.jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
