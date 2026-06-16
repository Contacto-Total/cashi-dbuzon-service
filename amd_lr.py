from __future__ import annotations
import os, math, json
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

try:
    import onnxruntime as _ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False
try:
    import librosa as _librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

SAMPLE_RATE = 8000
CHUNK_MS = 20
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class AMDConfig:
    sample_rate: int = SAMPLE_RATE
    chunk_ms: int = CHUNK_MS
    checkpoints_ms: List[int] = field(default_factory=lambda: [600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    persist_p_prev: float = 0.12
    model_path: str = os.path.join(_HERE, "funnel_model.json")
    silero_path: str = os.path.join(_HERE, "silero_vad.onnx")
    use_silero: bool = True
    energy_voiced_thr: float = 0.01


class Outcome(Enum):
    UNDECIDED = "UNDECIDED"
    BUZON = "BUZON"
    HUMANO = "HUMANO"
    SEND_TO_STT = "SEND_TO_STT"


@dataclass
class Decision:
    outcome: Outcome
    p_human: float
    checkpoint_ms: int
    reason: str = ""


class _SileroVAD:
    FRAME = 256

    def __init__(self, path: str):
        self.ok = False
        if _HAS_ORT and os.path.exists(path):
            try:
                _ort.set_default_logger_severity(3)
                self.sess = _ort.InferenceSession(path)
                self.ok = True
            except Exception:
                self.ok = False
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.voiced: List[bool] = []
        self._consumed = 0

    def update(self, buf: np.ndarray):
        if not self.ok:
            return
        while self._consumed + self.FRAME <= len(buf):
            x = buf[self._consumed:self._consumed + self.FRAME].astype(np.float32)[None, :]
            out, self._state = self.sess.run(
                None, {"input": x, "state": self._state, "sr": np.array(8000, dtype=np.int64)})
            self.voiced.append(bool(out[0, 0] > 0.5))
            self._consumed += self.FRAME

    def slope_at(self, t_ms: int) -> float:
        k = int(t_ms * SAMPLE_RATE / 256 / 1000.0)
        k = min(k, len(self.voiced))
        if k < 2:
            return 0.0
        h = k // 2
        v = np.array(self.voiced[:k], dtype=float)
        return float(v[h:].mean() - v[:h].mean())


def _energy_slope_at(buf: np.ndarray, t_ms: int, thr: float) -> float:
    n = SAMPLE_RATE * t_ms // 1000
    seg = buf[:n]; w = CHUNK_SAMPLES
    vs = [1.0 if np.sqrt(np.mean(seg[i:i + w] ** 2)) > thr else 0.0
          for i in range(0, len(seg) - w + 1, w)]
    if len(vs) < 2:
        return 0.0
    vs = np.array(vs); h = len(vs) // 2
    return float(vs[h:].mean() - vs[:h].mean())


def _g620_tick(seg: np.ndarray) -> float:
    w = seg[-480:] if len(seg) >= 480 else seg
    if len(w) < 32:
        return 0.0
    k = 620.0 / SAMPLE_RATE
    wr = 2 * math.cos(2 * math.pi * k)
    s1 = s2 = 0.0
    for x in w:
        s0 = x + wr * s1 - s2
        s2 = s1; s1 = s0
    power = s1 * s1 + s2 * s2 - wr * s1 * s2
    energy = float(np.sum(w * w)) + 1e-9
    return power / energy


def _f0_tick(seg: np.ndarray):
    w = seg[-800:] if len(seg) >= 800 else seg
    if len(w) < 200:
        return 0.0, 0.0, 0
    w = w - w.mean()
    ac = np.correlate(w, w, mode="full")[len(w) - 1:]
    if ac[0] <= 0:
        return 0.0, 0.0, 0
    ac = ac / ac[0]
    lo, hi = SAMPLE_RATE // 300, SAMPLE_RATE // 80
    seg_ac = ac[lo:hi]
    if len(seg_ac) == 0:
        return 0.0, 0.0, 0
    best = int(np.argmax(seg_ac)); strength = float(seg_ac[best])
    f0 = SAMPLE_RATE / (lo + best) if strength > 0.3 else 0.0
    npk = int(np.sum(seg_ac > 0.3))
    return f0, strength, npk


class _FunnelModel:
    def __init__(self, path: str):
        J = json.load(open(path))
        self.checkpoints = J["checkpoints_ms"]
        self.features = J["features"]
        self.models = J["models"]
        self.gates_buzon = J["gates_buzon"]
        self.gates_human = J["gates_human"]
        self.persist = J.get("persistencia_p_prev", 0.12)

    def p_human(self, ci: int, feat: np.ndarray) -> float:
        m = self.models[ci]
        z = m["intercept"]
        for k, c in enumerate(m["coef"]):
            z += c * (feat[k] - m["mu"][k]) / m["sd"][k]
        return 1.0 / (1.0 + math.exp(-z))


class StreamingAMD:
    def __init__(self, cfg: Optional[AMDConfig] = None):
        self.cfg = cfg or AMDConfig()
        self.model = _FunnelModel(self.cfg.model_path)
        self.checkpoints = self.model.checkpoints
        self._silero = _SileroVAD(self.cfg.silero_path) if self.cfg.use_silero else None
        self.using_silero = bool(self._silero and self._silero.ok)
        self.reset()

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_cp = 0
        self._prev_p = 1.0
        self._done = False
        if self._silero:
            self._silero.reset()
        self._tick_done = 0
        self._g620: List[float] = []
        self._f0: List[float] = []
        self._f0n: List[int] = []
        self._pitched: List[bool] = []

    @property
    def elapsed_ms(self) -> int:
        return len(self._buf) * 1000 // self.cfg.sample_rate

    def feed_chunk(self, chunk: np.ndarray) -> Decision:
        if self._done:
            return Decision(Outcome.UNDECIDED, self._prev_p, self.elapsed_ms)
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        self._buf = np.concatenate([self._buf, chunk])
        if self.using_silero:
            self._silero.update(self._buf)
        self._update_ticks()
        while self._next_cp < len(self.checkpoints) and \
                self.elapsed_ms >= self.checkpoints[self._next_cp]:
            d = self._decide_at(self._next_cp)
            self._next_cp += 1
            if d.outcome in (Outcome.BUZON, Outcome.HUMANO, Outcome.SEND_TO_STT):
                self._done = True
                return d
        return Decision(Outcome.UNDECIDED, self._prev_p, self.elapsed_ms)

    def force_decision(self) -> Decision:
        if self._done:
            return Decision(Outcome.UNDECIDED, self._prev_p, self.elapsed_ms)
        ci = min(self._next_cp, len(self.checkpoints) - 1)
        d = self._decide_at(ci, force=True)
        self._done = True
        return d

    def _update_ticks(self):
        total = len(self._buf) // CHUNK_SAMPLES
        while self._tick_done < total:
            end = (self._tick_done + 1) * CHUNK_SAMPLES
            seg = self._buf[:end]
            self._g620.append(_g620_tick(seg))
            f0, strg, npk = _f0_tick(seg)
            self._f0.append(f0); self._f0n.append(npk); self._pitched.append(f0 > 0)
            self._tick_done += 1

    def _feature_vector(self, ci: int) -> np.ndarray:
        t_ms = self.checkpoints[ci]
        n_tick = t_ms // self.cfg.chunk_ms
        g = np.array(self._g620[:n_tick]) if self._g620 else np.zeros(1)
        f0n = np.array(self._f0n[:n_tick]) if self._f0n else np.zeros(1)
        f0 = np.array(self._f0[:n_tick]) if self._f0 else np.zeros(1)
        pit = np.array(self._pitched[:n_tick]) if self._pitched else np.zeros(1, bool)
        f0n_rate = float((f0n >= 3).mean())
        g620 = float((g > 0.05).mean())
        if pit.sum() > 1:
            fp = f0[pit]
            roll = np.array([fp[max(0, j - 3):j + 1].std() for j in range(len(fp))])
            f0std_hi = float((roll >= 40).mean())
        else:
            f0std_hi = 0.0
        if self.using_silero:
            slope = self._silero.slope_at(t_ms)
        else:
            slope = _energy_slope_at(self._buf, t_ms, self.cfg.energy_voiced_thr)
        mfcc1 = mfcc3 = mfcc6 = rolloff = 0.0
        x = self._buf[:self.cfg.sample_rate * t_ms // 1000].astype(np.float32)
        if _HAS_LIBROSA and len(x) >= 512:
            mf = _librosa.feature.mfcc(y=x, sr=self.cfg.sample_rate, n_mfcc=13,
                                       n_fft=512, hop_length=160).mean(axis=1)
            mfcc1, mfcc3, mfcc6 = float(mf[1]), float(mf[3]), float(mf[6])
            rolloff = float(_librosa.feature.spectral_rolloff(
                y=x, sr=self.cfg.sample_rate, n_fft=512, hop_length=160).mean())
        return np.array([f0n_rate, g620, mfcc1, mfcc3, slope, rolloff, mfcc6, f0std_hi])

    def _decide_at(self, ci: int, force: bool = False) -> Decision:
        feat = self._feature_vector(ci)
        p = self.model.p_human(ci, feat)
        cp = self.checkpoints[ci]
        gB = self.model.gates_buzon[ci]; gH = self.model.gates_human[ci]
        out, reason = Outcome.UNDECIDED, ""
        if p >= gH:
            out, reason = Outcome.HUMANO, f"p={p:.2f}>=H({gH})"
        elif p <= gB and (ci == 0 or self._prev_p <= self.model.persist):
            out, reason = Outcome.BUZON, f"p={p:.2f}<=B({gB}) persistencia ok"
        if out is Outcome.UNDECIDED and (ci == len(self.checkpoints) - 1 or force):
            if p >= 0.5:
                reason = f"STT: tiende a humano (p={p:.2f})"
            elif p <= gB:
                reason = f"STT: maquina borderline (p={p:.2f}, persistencia lo freno)"
            else:
                reason = f"STT: zona media (p={p:.2f})"
            out = Outcome.SEND_TO_STT
        self._prev_p = p
        return Decision(out, p, cp, reason)


if __name__ == "__main__":
    import sys
    amd = StreamingAMD(AMDConfig())
    print(f"VAD: {'silero' if amd.using_silero else 'energia (fallback)'} | "
          f"librosa: {'si' if _HAS_LIBROSA else 'NO -> mfcc/rolloff=0, recalibra'}")
    if len(sys.argv) > 1:
        import soundfile as sf
        a, sr = sf.read(sys.argv[1], dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        for t in range(len(a) // CHUNK_SAMPLES):
            d = amd.feed_chunk(a[t * CHUNK_SAMPLES:(t + 1) * CHUNK_SAMPLES])
            if d.outcome is not Outcome.UNDECIDED:
                print(f"  -> {d.outcome.value} @ {d.checkpoint_ms}ms ({d.reason})"); break
        else:
            d = amd.force_decision()
            print(f"  -> {d.outcome.value} @ {d.checkpoint_ms}ms ({d.reason})")
    else:
        print("Uso: python amd_lr.py audio.wav")