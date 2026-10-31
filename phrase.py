"""
Phrase buffer: accumulate committed words, flush them as ONE utterance after a
period of stillness.

  committed word -> PhraseBuffer.add() ... (signer pauses) ... poll() -> words

"Still" means: no new word for `flush_after_s` AND the motion gate is not
mid-sign (pass busy=True while it is RECORDING, otherwise a long sign gets the
phrase cut in half).

SEAM FOR REORDERING: `reorder` sits between the buffer and speech. ASL gloss
order is not English order (e.g. topic-comment, time-first, no copula), so a
future step can map ["STORE", "I", "GO"] -> ["I", "go", "to", "the", "store"].
It is the identity today -- deliberately not implemented. Whatever replaces it
only has to be `list[str] -> list[str]`.
"""

from __future__ import annotations

import time
from typing import Callable

Reorder = Callable[[list[str]], list[str]]


def identity_reorder(words: list[str]) -> list[str]:
    return list(words)


class PhraseBuffer:
    def __init__(self, flush_after_s: float = 1.5,
                 reorder: Reorder = identity_reorder,
                 clock: Callable[[], float] = time.monotonic):
        if flush_after_s <= 0:
            raise ValueError("flush_after_s must be positive")
        self.flush_after_s = flush_after_s
        self.reorder = reorder
        self.clock = clock
        self._words: list[str] = []
        self._last_activity = 0.0

    def add(self, word: str, now: float | None = None) -> None:
        self._words.append(word)
        self._last_activity = self.clock() if now is None else now

    def poll(self, now: float | None = None, busy: bool = False) -> list[str] | None:
        """Returns the (reordered) utterance once, when stillness has elapsed."""
        now = self.clock() if now is None else now
        if busy:
            # Signer is mid-sign: that counts as activity.
            if self._words:
                self._last_activity = now
            return None
        if self._words and now - self._last_activity >= self.flush_after_s:
            out = self.reorder(self._words)
            self._words = []
            return out
        return None

    def flush(self) -> list[str] | None:
        """Force-flush (shutdown)."""
        if not self._words:
            return None
        out = self.reorder(self._words)
        self._words = []
        return out

    @property
    def pending(self) -> list[str]:
        return list(self._words)
