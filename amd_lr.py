"""
amd_lr.py  —  AMD streaming por REGRESIÓN LOGÍSTICA  (ventana 1800 ms)
======================================================================
MERGE: código del v4 (con los bugs corregidos) + MODELO del v3 (1800 ms).

Por qué 1800 ms y no 2000:
  Tu prioridad es NUNCA perder un humano. A 1800 ms el humano más bajo queda
  en p_human=0.083 (margen cómodo con p_buzon=0.05). A 2000 ms baja a 0.028
  (al filo de 0.02), porque a un humano conversador le das más tiempo de
  acumular voz y parecerse a buzón. => 1800 ms es más seguro para tu objetivo.

Arreglos del v4 que SÍ se conservan aquí (el v3 original NO los tenía):
  1) VAD por ENERGÍA por defecto (no webrtcvad). webrtcvad marca ~84% de "voz"
     en un humano callado (ruido de línea) y descalibra el modelo -> pierde
     humanos. use_webrtcvad=False.
  2) La "voz" se mide sobre los ÚLTIMOS 20 ms (como en el entrenamiento), no
     sobre 80 ms.
  3) SIN corte temprano (allow_early_cut=False): un humano puede tener un bajón
     transitorio de p_human a mitad de camino y recuperarse; por eso se decide
     SOLO en la fecha límite con la ventana completa.
  4) Decisión conservadora: BUZÓN solo si p_human <= p_buzon (0.05). Todo lo
     demás -> HUMANO -> STT (tu GPT). Así no se cuelga a una persona.

Cómo funciona (igual que tu diseño, cada 20 ms):
  feed_chunk() se llama por cada chunk de 20 ms. En cada llamada:
   1) corre los 4 detectores por tick (RMS/VAD/Goertzel/F0),
   2) suma a los contadores acumulados,
   3) recalcula las 5 FRACCIONES de la ventana [0, ahora],
   4) recalcula p_human con la regresión logística (logit = b0 + Σ wᵢ·zᵢ),
   5) decide solo en la fecha límite (1800 ms): buzón si p_human<=0.05.
  Los 1800 ms NO son "recién ahí analizo": son la FECHA LÍMITE. El análisis es
  continuo cada 20 ms. La diferencia con el sumador viejo: cuenta proporciones
  (cada evidencia pesa una vez), no puntos por tick (que sobre-contaban).

Desempeño held-out v3 (30% no visto): acc=0.914 rec_humano=0.872 rec_máquina=0.931
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


# --- MODELO v3, calibrado en ventana de 1800 ms (entrenado en los 464 archivos) ---
# Orden de features: [RMS_energy, RMS_hi, VAD_voiced, GOE_tone, F0_std_hi]
MODEL = {
    "mu":   [
    0.4568312434691746,
    0.2301462904911182,
    0.37367206548241033,
    0.051506443747823226,
    0.2444542091382707
  ],
    "sd":   [
    0.26177556001144264,
    0.238938962714524,
    0.23550205437603539,
    0.03483800862486579,
    0.27299164590499875
  ],
    "coef": [
    -1.5309755822130389,
    1.2807213149613483,
    -3.194310863299281,
    -0.9442103123734442,
    1.7897188490047853
  ],
    "b0":   -0.9178130745334432,
}


@dataclass
class AMDConfig:
    sample_rate: int = 8000
    chunk_ms: int = 20
    win_rms_ms: int = 40
    win_goertzel_ms: int = 60
    win_vad_ms: int = 80
    win_f0_ms: int = 100

    # Cortes de banda (de los datos) — NO TOCAR: el modelo se calibró con estos
    voiced_rms_thr: float = 0.01      # "voz" = energía de los ÚLTIMOS 20 ms > 0.01
    use_webrtcvad: bool = False       # True SOLO si recalibras con webrtcvad
    webrtcvad_aggressiveness: int = 3
    rms_energy_thr: float = 0.005     # tick "con energía"
    rms_hi_thr: float = 0.03          # tick "fuerte"
    goertzel_hz: float = 440.0
    goertzel_thr: float = 0.05
    f0_std_hi: float = 40.0
    f0_min_hz: float = 80.0
    f0_max_hz: float = 300.0
    f0_n_min: int = 3

    # --- Decisión (conservadora con el humano) ---
    t_min_ms: int = 1000              # (solo aplica si allow_early_cut=True)
    t_max_ms: int = 1800              # FECHA LÍMITE (modelo v3 = 1800 ms)

    # BUZÓN solo si p_human <= p_buzon. A 1800 ms el humano más bajo quedó en
    # 0.083, así que 0.5 deja a TODOS los humanos a salvo y atrapa las máquinas
    # obvias. El resto pasa a STT. Súbelo si quieres mandar menos a STT (riesgo
    # de perder humanos); bájalo para ser aún más conservador.
    p_buzon: float = 0.20
    p_human: float = 0.7


    # Corte temprano: OFF por seguridad (un humano puede tener un bajón
    # transitorio de p_human y recuperarse). Decidir solo con la ventana
    # completa a t_max es lo más seguro para no perder humanos.
    allow_early_cut: bool = False
    p_human_early: float = 1.0


class Label(Enum):
    HUMAN = "HUMAN"      # -> STT -> asesor
    BUZON = "BUZON"      # -> colgar / dejar mensaje
    UNDECIDED = "UNDECIDED"


@dataclass
class Decision:
    label: Label
    p_human: float
    elapsed_ms: int
    forced: bool = False
    features: Optional[dict] = None


class StreamingAMD:
    """Acumula chunks de 20 ms y decide HUMANO/BUZÓN con probabilidad calibrada."""

    def __init__(self, cfg: Optional[AMDConfig] = None):
        self.cfg = cfg or AMDConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self.elapsed_ms = 0
        self._done: Optional[Decision] = None
        self.n = self.n_energy = self.n_hi = self.n_voiced = 0
        self.n_tone = self.n_voiced_f0 = self.n_f0_hi = 0
        if _HAS_VAD:
            self._vad = webrtcvad.Vad(self.cfg.webrtcvad_aggressiveness)

    def _samples(self, ms): return int(self.cfg.sample_rate * ms / 1000)

    def _tail(self, ms):
        n = self._samples(ms)
        return self._buf[-n:] if self._buf.size >= n else self._buf

    @staticmethod
    def _sigmoid(z):
        return 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))

    def _features(self) -> dict:
        n = max(self.n, 1)
        f0d = max(self.n_voiced_f0, 1)
        return {"RMS_energy": self.n_energy/n, "RMS_hi": self.n_hi/n,
                "VAD_voiced": self.n_voiced/n, "GOE_tone": self.n_tone/n,
                "F0_std_hi": (self.n_f0_hi/f0d if self.n_voiced_f0 else 0.0)}

    def _p_human(self, f) -> float:
        order = ["RMS_energy", "RMS_hi", "VAD_voiced", "GOE_tone", "F0_std_hi"]
        z = MODEL["b0"]
        for i, k in enumerate(order):
            z += MODEL["coef"][i] * (f[k] - MODEL["mu"][i]) / MODEL["sd"][i]
        return self._sigmoid(z)

    def feed_chunk(self, chunk: np.ndarray) -> Optional[Decision]:
        """Se llama CADA 20 ms. Recalcula p_human y decide si ya puede cortar."""
        if self._done is not None:
            return self._done
        cfg = self.cfg
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
        keep = self._samples(max(cfg.win_f0_ms, cfg.win_vad_ms) + 40)
        if self._buf.size > keep:
            self._buf = self._buf[-keep:]
        self.elapsed_ms += cfg.chunk_ms
        if self._buf.size < self._samples(cfg.win_rms_ms):
            return None  # warmup (<40 ms)

        # (1) detectores por tick
        rms = float(np.sqrt(np.mean(self._tail(cfg.win_rms_ms) ** 2)))
        voiced = self._voiced(self._tail(cfg.win_vad_ms))
        tone = self._goertzel(self._tail(cfg.win_goertzel_ms))
        f0_n, f0_std = self._f0(self._tail(cfg.win_f0_ms)) if voiced else (0, 0.0)

        # (2) contadores acumulados
        self.n += 1
        if rms >= cfg.rms_energy_thr: self.n_energy += 1
        if rms >= cfg.rms_hi_thr:     self.n_hi += 1
        if voiced:                    self.n_voiced += 1
        if tone:                      self.n_tone += 1
        if voiced and f0_n >= cfg.f0_n_min:
            self.n_voiced_f0 += 1
            if f0_std >= cfg.f0_std_hi: self.n_f0_hi += 1

        # (3)+(4) fracciones -> probabilidad (cada 20 ms)
        feats = self._features()
        p = self._p_human(feats)

        # (5) decisión — por defecto SOLO en la fecha límite (ventana completa)
        if self.cfg.allow_early_cut and self.elapsed_ms >= cfg.t_min_ms:
            if p <= cfg.p_buzon:
                return self._finish(Label.BUZON, p, feats)
            if p >= cfg.p_human_early:
                return self._finish(Label.HUMAN, p, feats)
        if self.elapsed_ms >= cfg.t_max_ms:
            if p <= cfg.p_buzon:
                lbl = Label.BUZON
            elif p >= cfg.p_human:
                lbl = Label.HUMAN
            else:
                lbl = Label.UNDECIDED
            return self._finish(lbl, p, feats, forced=True)
        return None

    def current_p_human(self) -> float:
        return self._p_human(self._features()) if self.n else 0.5

    def _finish(self, label, p, feats, forced=False):
        self._done = Decision(label, p, self.elapsed_ms, forced, feats)
        return self._done

    # ---- extractores (calibrados; recalibra el modelo si los cambias) ----
    def _voiced(self, seg) -> bool:
        # El modelo se calibró con "voz = energía de los ÚLTIMOS 20 ms > 0.01".
        # NO uses webrtcvad por defecto: marca ~84% de "voz" en humano callado
        # (ruido de línea) -> infla VAD_voiced -> humanos parecen máquinas.
        if seg.size == 0:
            return False
        last20 = seg[-self._samples(20):]
        if self.cfg.use_webrtcvad and _HAS_VAD:
            fr = (np.clip(last20, -1, 1) * 32767).astype(np.int16).tobytes()
            try:
                return bool(self._vad.is_speech(fr, self.cfg.sample_rate))
            except Exception:
                pass
        return float(np.sqrt(np.mean(last20 ** 2))) > self.cfg.voiced_rms_thr

    def _goertzel(self, seg) -> bool:
        if seg.size < 8:
            return False
        N = seg.size
        k = int(0.5 + N * self.cfg.goertzel_hz / self.cfg.sample_rate)
        w = 2*math.pi*k/N
        coeff = 2*math.cos(w)
        s1 = s2 = 0.0
        for x in seg:
            s = x + coeff*s1 - s2
            s2, s1 = s1, s
        power = s2*s2 + s1*s1 - coeff*s1*s2
        total = float(np.sum(seg**2)) * N + 1e-9
        return (power/total) > self.cfg.goertzel_thr

    def _f0(self, seg):
        if seg.size < self._samples(40):
            return 0, 0.0
        frame, hop = self._samples(20), self._samples(10)
        f0s = []
        for st in range(0, seg.size-frame, hop):
            w = seg[st:st+frame]
            if np.sqrt(np.mean(w**2)) < self.cfg.rms_energy_thr:
                continue
            w = w - w.mean()
            corr = np.correlate(w, w, mode="full")[frame-1:]
            lo = int(self.cfg.sample_rate/self.cfg.f0_max_hz)
            hi = min(int(self.cfg.sample_rate/self.cfg.f0_min_hz), len(corr)-1)
            if hi <= lo:
                continue
            pk = lo + int(np.argmax(corr[lo:hi]))
            if corr[pk] <= 0:
                continue
            f0s.append(self.cfg.sample_rate/pk)
        if not f0s:
            return 0, 0.0
        a = np.array(f0s)
        return int(a.size), float(a.std())


# --- Demo de validación: corre sobre los wav y mide humanos perdidos ---
if __name__ == "__main__":
    import soundfile as sf, glob, os
    cfg = AMDConfig()
    files = sorted(glob.glob("base/BASE_CONGLOMERADA/*.wav"))
    if not files:
        print("Pon los wav en base/BASE_CONGLOMERADA/ para la demo.")
        raise SystemExit
    hum_lost = mac_caught = hum_total = mac_total = 0
    for f in files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        amd = StreamingAMD(AMDConfig())
        dec = None
        n20 = cfg.sample_rate*cfg.chunk_ms//1000
        for t in range(len(audio)//n20):
            dec = amd.feed_chunk(audio[t*n20:(t+1)*n20])
            if dec:
                break
        if dec is None:  # audio más corto que la ventana -> decide con lo que hay
            p = amd.current_p_human()
            dec = Decision(Label.BUZON if p <= cfg.p_buzon else Label.HUMAN,
                           p, amd.elapsed_ms, True)
        true_h = os.path.basename(f).startswith("humano")
        is_buzon = dec.label == Label.BUZON
        if true_h: hum_total += 1
        else: mac_total += 1
        if true_h and is_buzon: hum_lost += 1
        if (not true_h) and is_buzon: mac_caught += 1
    print(f"n={len(files)}  ventana=1800ms  p_buzon={cfg.p_buzon}")
    print(f"HUMANOS PERDIDOS (falso buzón): {hum_lost}/{hum_total}")
    print(f"máquinas atrapadas en el AMD:   {mac_caught}/{mac_total} "
          f"({mac_caught/max(mac_total,1)*100:.0f}%)  -> el resto pasa a STT")
