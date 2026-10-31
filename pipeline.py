"""
Three-stage threading: capture -> inference -> audio, bounded queues, and the
rule that CAPTURE NEVER BLOCKS (on inference or on playback).

  capture thread   reads camera, drops the OLDEST queued frame when inference is
                   behind (a stale frame is worth less than a fresh one).
  inference thread MediaPipe + gate + model + admission + phrase buffer.
  audio thread     plays WAVs; inference hands off with put-nowait and moves on.

The queue helpers and Speaker are separate from the camera so they can be
exercised with synthetic input (test_pipeline.py).
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable


class DropStats:
    def __init__(self):
        self.dropped = 0
        self.put = 0


def put_drop_oldest(q: Queue, item, stats: DropStats | None = None) -> bool:
    """Never blocks. If full, discards the oldest item. Returns True if one was dropped."""
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            break
        except Full:
            try:
                q.get_nowait()
                dropped = True
            except Empty:
                pass
    if stats:
        stats.put += 1
        stats.dropped += int(dropped)
    return dropped


def put_drop_newest(q: Queue, item, stats: DropStats | None = None) -> bool:
    """Never blocks. If full, discards `item` itself. Returns True if dropped."""
    try:
        q.put_nowait(item)
        if stats:
            stats.put += 1
        return False
    except Full:
        if stats:
            stats.put += 1
            stats.dropped += 1
        return True


class Speaker:
    """
    Plays a pre-rendered WAV per word (see scripts/render_wavs.sh). `say_phrase`
    enqueues a whole utterance; playback happens on its own thread. The queue is
    bounded and put is non-blocking: if audio falls far behind, the newest
    utterance is dropped (and counted) rather than stalling the camera.
    """

    def __init__(self, wav_dir: Path, player: str = "aplay", maxsize: int = 8,
                 play_fn: Callable[[Path], None] | None = None):
        self.wav_dir = Path(wav_dir)
        self.player = player
        self.q: Queue[list[str]] = Queue(maxsize=maxsize)
        self.stats = DropStats()
        self.spoken: list[str] = []
        self._play = play_fn or self._play_wav
        self._stop = threading.Event()
        self._warned_player = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio")
        self._thread.start()

    def say(self, word: str) -> bool:
        return self.say_phrase([word])

    def say_phrase(self, words: list[str]) -> bool:
        return not put_drop_newest(self.q, list(words), self.stats)

    def _play_wav(self, wav: Path) -> None:
        try:
            subprocess.run([self.player, "-q", str(wav)], check=False, capture_output=True)
        except FileNotFoundError:
            if not self._warned_player:
                print(f"[audio] player '{self.player}' not found")
                self._warned_player = True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                words = self.q.get(timeout=0.1)
            except Empty:
                continue
            for w in words:
                wav = self.wav_dir / f"{w}.wav"
                if not wav.exists():
                    print(f"[audio] missing {wav.name}")      # skip, never crash
                    continue
                self._play(wav)
                self.spoken.append(w)

    def close(self, drain_s: float = 0.0) -> None:
        end = time.monotonic() + drain_s
        while drain_s and not self.q.empty() and time.monotonic() < end:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=2.0)
