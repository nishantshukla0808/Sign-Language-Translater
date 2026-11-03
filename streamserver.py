"""Tiny MJPEG server: lets you watch the annotated camera feed in a browser (Pi with no display)."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class StreamServer:
    def __init__(self, port: int, button: str | None = None):
        """button: label of an optional Start button on the page; clicking it sets self.started."""
        self.jpg: bytes | None = None
        self.started = threading.Event()
        if button is None:
            self.started.set()
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
                    btn = ("<button onclick=\"fetch('/go',{method:'POST'});this.disabled=true;this.textContent='running...'\" "
                           "style='font-size:22px;padding:14px 28px;margin:10px;border:0;border-radius:10px;"
                           f"background:#2e8b57;color:#fff'>{button}</button>" if button else "")
                    body = ("<html><body style='margin:0;background:#111'>" + btn +
                            "<img src='/stream' style='width:100%;max-width:1100px;display:block'></body></html>").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)

        outer_started = self.started

        def do_POST(self):
            outer_started.set()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        H.do_POST = do_POST
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="stream").start()

    def publish(self, bgr) -> None:
        self.jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
