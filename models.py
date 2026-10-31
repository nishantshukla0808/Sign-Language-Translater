"""Model selection shared by realtime.py and eval_clips.py (--model {popsign,aslcitizen})."""

from __future__ import annotations

from pathlib import Path

CHOICES = ("popsign", "aslcitizen")

DEFAULTS = {
    "popsign": {"model": "models/asl-signs.onnx",
                "labels": "models/sign_to_prediction_index_map.json"},
    "aslcitizen": {"model": "models/aslcitizen_stgcn.onnx",
                   "labels": "models/aslcitizen_gloss.json"},
}


def load_model(name: str, model_path: str | None = None, labels_path: str | None = None,
               threads: int = 2, aslcitizen_norm: str = "train"):
    d = DEFAULTS[name]
    model_path = model_path or d["model"]
    labels_path = labels_path or d["labels"]
    if not Path(model_path).exists():
        hint = ("run scripts/fetch_models.sh" if name == "popsign"
                else "run `python export_stgcn.py` (needs PyTorch once) to create it")
        raise SystemExit(f"model file not found: {model_path}  ({hint})")
    if name == "popsign":
        from classifier import SignClassifier
        return SignClassifier(model_path, labels_path, threads=threads)
    from aslcitizen import AslCitizenClassifier
    return AslCitizenClassifier(model_path, labels_path, threads=threads, norm=aslcitizen_norm)


def _norm(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


class RestrictedModel:
    """
    Wraps a classifier so that only an allowed set of words can win.

    Everything else is zeroed and the rest renormalised to sum to 1. A word like
    HOW matches the model's HOW1 / HOW2 variants. Caveat: renormalising makes the
    scores look more confident than they are -- with 4 allowed words the model
    ALWAYS picks one of them, so the reject floor stops meaning "unsure".
    """

    def __init__(self, inner, words: list[str]):
        import numpy as np
        self._np = np
        self.inner = inner
        self.name = inner.name
        self.vocab_size = inner.vocab_size
        self.words = [_norm(w) for w in words if _norm(w)]
        idx, self.unmatched, self.matched = [], [], {}
        for w in self.words:
            hit = [i for i, lab in inner.idx_to_sign.items()
                   if _norm(lab) == w or (_norm(lab).startswith(w) and _norm(lab)[len(w):].isdigit())]
            if hit:
                idx += hit
                self.matched[w] = [inner.idx_to_sign[i] for i in hit]
            else:
                self.unmatched.append(w)
        if not idx:
            raise SystemExit(f"--vocab: none of {words} exist in the {inner.name} vocabulary")
        self.mask = np.zeros(inner.vocab_size, dtype=bool)
        self.mask[sorted(set(idx))] = True

    def __getattr__(self, k):                  # idx_to_sign, labels_are_real, ...
        return getattr(self.inner, k)

    def predict(self, window):
        p = self.inner.predict(window) * self.mask
        s = p.sum()
        return p / s if s > 0 else self.mask / self.mask.sum()

    def top_k(self, probs, k: int = 2):
        return self.inner.top_k(probs, k)

    def warmup(self, frames: int = 32) -> float:
        return self.inner.warmup(frames)
