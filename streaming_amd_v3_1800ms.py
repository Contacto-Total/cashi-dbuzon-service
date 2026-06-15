"""
streaming_amd_v3.py
===================
AMD en streaming, CALIBRADO con BASE_CONGLOMERADA (130 humano / 334 máquina).

Cambio clave respecto a v2: el "score" ya NO es una suma de puntos por tick
(eso sobre-contaba evidencia: dos ticks de silencio seguidos son casi el mismo
dato y sumaban doble, lo que mandaba todo a HUMANO). Ahora el score es una
REGRESIÓN LOGÍSTICA sobre features AGREGADAS de la ventana acumulada [0, t]:
cada evidencia se cuenta una sola vez, la salida es una probabilidad calibrada
y sigue siendo un score aditivo (logit = b0 + Σ wᵢ·zᵢ).

Lo que dijeron los datos (y corrige supuestos de v2):
  * VAD/voz: ACTIVIDAD continua => MÁQUINA. El humano contesta corto y calla
    (frac. de voz: humano≈0.12, máquina≈0.45). VAD_voiced es la señal dominante.
  * RMS energía: más energía => máquina (mismo motivo).
  * Goertzel: mejor banda ≈ 440 Hz (tono de progreso), señal DÉBIL pero útil.
  * F0_std: alta variación de pitch tiende a máquina (música/TTS/ruido).
  * Honestidad temporal: dentro de 1800 ms la detección recién se vuelve fiable
    hacia 1200–1800 ms (antes la máquina aún no arrancó su saludo). Forzar
    decisión a 300–500 ms hunde el acierto. Por eso se decide tarde y solo se
    sale antes ante confianza extrema.

Desempeño held-out (30% no visto): acc=0.914  rec_humano=0.872  rec_máquina=0.931
  falso_buzón=5  falso_humano=7  (n=140).

Enchufa tus extractores reales en los  # >>> PLUG.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

try:
    import webrtcvad
    _HAS_VAD = True
except Exception:
    _HAS_VAD = False


# --- Modelo calibrado (entrenado en los 464 archivos) ----------------------
# Orden de features: [RMS_energy, RMS_hi, VAD_voiced, GOE_tone, F0_std_hi]
MODEL = {
    "mu":    [0.4401, 0.2246, 0.3599, 0.0507, 0.2401],
    "sd":    [0.2552, 0.2349, 0.2307, 0.0329, 0.2732],
    "coef":  [-1.544, 1.427, -3.016, -0.797, 1.433],
    "b0":    -0.752,
}


@dataclass
class AMDConfig:
    sample_rate: int = 8000
    chunk_ms: int = 20
    win_rms_ms: int = 40
    win_goertzel_ms: int = 60
    win_vad_ms: int = 80
    win_f0_ms: int = 100

    # Cortes de banda (de los datos)
    rms_energy_thr: float = 0.005     # tick "con energía"
    rms_hi_thr: float = 0.03          # tick "fuerte"
    goertzel_hz: float = 440.0        # banda elegida por separación
    goertzel_thr: float = 0.05        # frac. de energía en banda -> "tono"
    f0_std_hi: float = 40.0           # f0_std alto
    f0_min_hz: float = 80.0
    f0_max_hz: float = 300.0
    f0_n_min: int = 3

    # Decisión
    t_min_ms: int = 1000              # no decidir antes (la señal aún no madura)
    t_max_ms: int = 1800              # ventana de decisión
    # Umbral de probabilidad para la decisión final en t_max.
    #   0.5 = balanceado.  Súbelo (p.ej. 0.6) para declarar MÁQUINA solo con más
    #   evidencia y reducir falso_buzón (colgar a una persona), a costa de gastar
    #   más STT con máquinas.
    p_decide: float = 0.5
    # Salida temprana solo ante confianza extrema (las decisiones tempranas son
    # poco fiables según los datos; por eso bandas muy estrictas).
    p_early_human: float = 0.98
    p_early_machine: float = 0.02


class Label(Enum):
    HUMAN = "HUMAN"
    BUZON = "BUZON"
    UNDECIDED = "UNDECIDED"


@dataclass
class Decision:
    label: Label
    p_human: float
    elapsed_ms: int
    forced: bool = False
    features: Optional[dict] = None


class StreamingAMD:
    """Acumula chunks de 20 ms y decide HUMANO/BUZÓN con prob. calibrada."""

    def __init__(self, cfg: Optional[AMDConfig] = None):
        self.cfg = cfg or AMDConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self.elapsed_ms = 0
        self._done: Optional[Decision] = None
        # Contadores acumulados de la ventana
        self.n = 0
        self.n_energy = 0
        self.n_hi = 0
        self.n_voiced = 0
        self.n_tone = 0
        self.n_voiced_f0 = 0   # ticks con voz y f0 fiable
        self.n_f0_hi = 0       # de esos, con f0_std alto
        if _HAS_VAD:
            self._vad = webrtcvad.Vad(2)

    # ---- utilidades de ventana ----
    def _samples(self, ms): return int(self.cfg.sample_rate * ms / 1000)
    def _tail(self, ms):
        n = self._samples(ms)
        return self._buf[-n:] if self._buf.size >= n else self._buf

    @staticmethod
    def _sigmoid(z): return 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))

    def _features(self) -> dict:
        n = max(self.n, 1)
        f0_den = max(self.n_voiced_f0, 1)
        return {
            "RMS_energy": self.n_energy / n,
            "RMS_hi":     self.n_hi / n,
            "VAD_voiced": self.n_voiced / n,
            "GOE_tone":   self.n_tone / n,
            "F0_std_hi":  self.n_f0_hi / f0_den if self.n_voiced_f0 else 0.0,
        }

    def _p_human(self, feats: dict) -> float:
        order = ["RMS_energy", "RMS_hi", "VAD_voiced", "GOE_tone", "F0_std_hi"]
        z = MODEL["b0"]
        for i, k in enumerate(order):
            z += MODEL["coef"][i] * (feats[k] - MODEL["mu"][i]) / MODEL["sd"][i]
        return self._sigmoid(z)

    # ---- entrada de audio ----
    def feed_chunk(self, chunk: np.ndarray) -> Optional[Decision]:
        """chunk: 20 ms float32 mono en [-1,1]. Devuelve Decision o None."""
        if self._done is not None:
            return self._done
        cfg = self.cfg
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
        keep = self._samples(max(cfg.win_f0_ms, cfg.win_vad_ms) + 40)
        if self._buf.size > keep:
            self._buf = self._buf[-keep:]
        self.elapsed_ms += cfg.chunk_ms

        if self._buf.size < self._samples(cfg.win_rms_ms):
            return None  # warmup

        # --- features por tick (>>> PLUG: enchufa tus extractores reales) ---
        rms = float(np.sqrt(np.mean(self._tail(cfg.win_rms_ms) ** 2)))
        voiced = self._voiced(self._tail(cfg.win_vad_ms))
        tone = self._goertzel(self._tail(cfg.win_goertzel_ms))
        f0_n, f0_std = self._f0(self._tail(cfg.win_f0_ms)) if voiced else (0, 0.0)

        # --- acumular contadores ---
        self.n += 1
        if rms >= cfg.rms_energy_thr: self.n_energy += 1
        if rms >= cfg.rms_hi_thr:     self.n_hi += 1
        if voiced:                    self.n_voiced += 1
        if tone:                      self.n_tone += 1
        if voiced and f0_n >= cfg.f0_n_min:
            self.n_voiced_f0 += 1
            if f0_std >= cfg.f0_std_hi: self.n_f0_hi += 1

        feats = self._features()
        p = self._p_human(feats)

        # --- decisión ---
        if self.elapsed_ms >= cfg.t_min_ms:
            if p >= cfg.p_early_human:
                return self._finish(Label.HUMAN, p, feats)
            if p <= cfg.p_early_machine:
                return self._finish(Label.BUZON, p, feats)
        if self.elapsed_ms >= cfg.t_max_ms:
            lbl = Label.HUMAN if p >= cfg.p_decide else Label.BUZON
            return self._finish(lbl, p, feats, forced=True)
        return None

    def current_p_human(self) -> float:
        return self._p_human(self._features()) if self.n else 0.5

    def _finish(self, label, p, feats, forced=False):
        self._done = Decision(label, p, self.elapsed_ms, forced, feats)
        return self._done

    # ---- stubs de extracción (reemplaza por tus modelos reales) ----
    def _voiced(self, seg) -> bool:
        if seg.size == 0: return False
        if _HAS_VAD:
            fr = (np.clip(seg[-self._samples(20):], -1, 1) * 32767).astype(np.int16).tobytes()
            try: return bool(self._vad.is_speech(fr, self.cfg.sample_rate))
            except Exception: pass
        return float(np.sqrt(np.mean(seg ** 2))) > self.cfg.rms_energy_thr

    def _goertzel(self, seg) -> bool:
        if seg.size < 8: return False
        N = seg.size
        k = int(0.5 + N * self.cfg.goertzel_hz / self.cfg.sample_rate)
        w = 2*math.pi*k/N; coeff = 2*math.cos(w); s1 = s2 = 0.0
        for x in seg:
            s = x + coeff*s1 - s2; s2, s1 = s1, s
        power = s2*s2 + s1*s1 - coeff*s1*s2
        total = float(np.sum(seg**2)) * N + 1e-9     # normalización de Parseval
        return (power/total) > self.cfg.goertzel_thr

    def _f0(self, seg):
        if seg.size < self._samples(40): return 0, 0.0
        frame, hop = self._samples(20), self._samples(10); f0s = []
        for st in range(0, seg.size-frame, hop):
            w = seg[st:st+frame]
            if np.sqrt(np.mean(w**2)) < self.cfg.rms_energy_thr: continue
            w = w - w.mean()
            corr = np.correlate(w, w, mode="full")[frame-1:]
            lo = int(self.cfg.sample_rate/self.cfg.f0_max_hz)
            hi = min(int(self.cfg.sample_rate/self.cfg.f0_min_hz), len(corr)-1)
            if hi <= lo: continue
            pk = lo + int(np.argmax(corr[lo:hi]))
            if corr[pk] <= 0: continue
            f0s.append(self.cfg.sample_rate/pk)
        if not f0s: return 0, 0.0
        a = np.array(f0s); return int(a.size), float(a.std())


if __name__ == "__main__":
    import soundfile as sf, glob, os
    cfg = AMDConfig()
    files = sorted(glob.glob("base/BASE_CONGLOMERADA/*.wav"))
    if not files:
        print("Coloca los wav en base/BASE_CONGLOMERADA/ para la demo."); raise SystemExit
    tp=tn=fp=fn=0
    for f in files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        amd = StreamingAMD(AMDConfig()); dec = None
        n20 = cfg.sample_rate*cfg.chunk_ms//1000
        for t in range(len(audio)//n20):
            dec = amd.feed_chunk(audio[t*n20:(t+1)*n20])
            if dec: break
        if dec is None:
            p = amd.current_p_human(); dec = Decision(Label.HUMAN if p>=0.5 else Label.BUZON, p, amd.elapsed_ms, True)
        true_h = os.path.basename(f).startswith("humano")
        pred_h = dec.label == Label.HUMAN
        if true_h and pred_h: tp+=1
        elif (not true_h) and (not pred_h): tn+=1
        elif (not true_h) and pred_h: fp+=1
        else: fn+=1
    n=len(files)
    print(f"n={n}  acc={(tp+tn)/n:.3f}  rec_humano={tp/(tp+fn):.3f}  rec_maquina={tn/(tn+fp):.3f}")
    print(f"falso_buzon={fn}  falso_humano={fp}")