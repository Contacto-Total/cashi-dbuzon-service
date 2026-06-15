"""
streaming_amd_v4.py
===================
AMD en streaming — ventana 2000 ms, calibrado con BASE_CONGLOMERADA.
Prioridad de negocio: NUNCA perder un humano (que llegue al asesor vía STT).
El costo de STT no importa; sí importa no colgarle a una persona.

==========================================================================
¿SIGUE ESCUCHANDO CADA 20 ms?  -> SÍ.  Lee esto:
==========================================================================
feed_chunk() se llama UNA VEZ POR CADA CHUNK DE 20 ms. En CADA llamada:
  1) actualiza los 4 detectores por tick (RMS/VAD/Goertzel/F0) sobre sus
     ventanas (últimos 40/60/80/100 ms),
  2) suma 1 a los contadores acumulados (cuántos ticks fueron voz, energía...),
  3) RECALCULA las 5 fracciones sobre [0, ahora],
  4) RECALCULA p_human con la regresión logística,
  5) decide si ya puede cortar.
O sea: el análisis es CONTINUO, cada 20 ms, igual que tu diseño original.
Los 2000 ms NO son "esperar y recién ahí analizar": son la FECHA LÍMITE.

En la fecha límite se clasifica en TRES destinos (tu esquema de dos umbrales
+ incertidumbre):
  p_human >= 0.98  -> HUMANO  : directo al asesor (se salta el STT)
  p_human <= 0.02  -> BUZÓN   : colgar / dejar mensaje
  en medio         -> STT     : incertidumbre, lo deciden STT + LLM
(Nota: tu buzón original era "no-humano >= 95%" = p_human <= 0.05, pero a 0.05
se pierde 1 humano; por eso el buzón quedó en 0.02 ≈ 98% de confianza.)

Lo único que cambió respecto a tu sumador de puntos: el paso (3)-(4). Antes
sumabas puntos por tick (y dos ticks de silencio sumaban doble). Ahora cuentas
PROPORCIONES de la ventana y las pasas por la logística -> cada evidencia pesa
una sola vez y la salida es una probabilidad calibrada.

Desempeño held-out 2000 ms: acc=0.921 rec_humano=0.846 rec_máquina=0.950.
Nota: extender de 1800->2000 ms sube el recall de máquina, PERO un humano
"conversador" puede acumular voz y parecer máquina. Por eso el umbral de buzón
es MUY conservador (solo corta buzón con p_human <= p_buzon).
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


# --- Modelo calibrado en ventana de 2000 ms (entrenado en los 464 archivos) ---
# Orden: [RMS_energy, RMS_hi, VAD_voiced, GOE_tone, F0_std_hi]
# Precisión COMPLETA, idéntica a amd_final_model.json (la fuente de verdad).
# No redondear: si editas estos números, edita también el JSON (y viceversa).
MODEL = {
    "mu":   [0.4568312434691746, 0.2301462904911182, 0.37367206548241033,
             0.051506443747823226, 0.2444542091382707],
    "sd":   [0.26177556001144264, 0.238938962714524, 0.23550205437603539,
             0.03483800862486579, 0.27299164590499875],
    "coef": [-1.5309755822130389, 1.2807213149613483, -3.194310863299281,
             -0.9442103123734442, 1.7897188490047853],
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

    # Cortes de banda (de los datos)
    voiced_rms_thr: float = 0.01      # VAD por energía (con lo que se ENTRENÓ)
    use_webrtcvad: bool = False       # True solo si recalibras con webrtcvad
    webrtcvad_aggressiveness: int = 3
    rms_energy_thr: float = 0.005     # tick "con energía"
    rms_hi_thr: float = 0.03          # tick "fuerte"
    goertzel_hz: float = 440.0
    goertzel_thr: float = 0.05
    f0_std_hi: float = 40.0
    f0_min_hz: float = 80.0
    f0_max_hz: float = 300.0
    f0_n_min: int = 3

    # --- Decisión (asimétrica, conservadora con el humano) ---
    t_min_ms: int = 800               # no cortar buzón antes (early = poco fiable)
    t_max_ms: int = 2000              # FECHA LÍMITE: aquí se fuerza el hand-off

    # --- TRES CUBOS (tu diseño original: dos umbrales + incertidumbre) ---
    # En la fecha límite se clasifica en uno de tres destinos:
    #   p_human <= p_buzon          -> BUZÓN  (colgar / voicemail)
    #   p_human >= p_human_decide   -> HUMANO (directo al asesor, se salta STT)
    #   en medio (incertidumbre)    -> STT    (STT + LLM deciden)
    #
    # p_human_decide = 0.98 es tu "confianza humano >= 98%". Con los datos, solo
    # 1 máquina supera 0.98 (llega al asesor directo); 69 humanos pasan directo.
    #
    # OJO con p_buzon: tu idea original era "no-humano >= 95%" (p_human <= 0.05),
    # pero a 0.05 se PIERDE 1 humano. Por eso lo dejo en 0.02 (equivale a exigir
    # ~98% de confianza también para el buzón): 0 humanos perdidos en los 464
    # archivos (el humano más bajo quedó en 0.028) y aún atrapa ~22% de máquinas.
    p_buzon: float = 0.02
    p_human_decide: float = 0.98

    # Cortes tempranos: por defecto OFF. Cortar buzón antes de la fecha límite
    # pierde humanos, porque un humano puede tener un bajón TRANSITORIO de
    # p_human a mitad de camino (una ráfaga de voz) y recuperarse luego. Decidir
    # solo con la ventana completa a 2000 ms es lo más seguro para no perder
    # humanos. Ponlo en True solo si la latencia de cortar máquinas te importa
    # más que perder algún humano (no es tu caso).
    allow_early_cut: bool = False


class Label(Enum):
    HUMAN = "HUMAN"      # p_human >= p_human_decide -> ASESOR DIRECTO (se salta STT)
    STT   = "STT"        # zona de incertidumbre  -> STT + LLM (segunda etapa)
    BUZON = "BUZON"      # p_human <= p_buzon      -> colgar / dejar mensaje
    UNDECIDED = "UNDECIDED"   # aún sin suficiente audio / antes de la fecha límite


@dataclass
class Decision:
    label: Label
    p_human: float
    elapsed_ms: int
    forced: bool = False        # True si se resolvió en la fecha límite
    features: Optional[dict] = None


class StreamingAMD:
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
    def _sigmoid(z): return 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))

    def _features(self) -> dict:
        n = max(self.n, 1); f0d = max(self.n_voiced_f0, 1)
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
        if self._buf.size > keep: self._buf = self._buf[-keep:]
        self.elapsed_ms += cfg.chunk_ms
        if self._buf.size < self._samples(cfg.win_rms_ms):
            return None  # warmup (<40 ms)

        # (1) detectores por tick  [>>> PLUG: tus extractores reales]
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

        # (5) decisión — TRES CUBOS (tu diseño): buzón / humano / incertidumbre
        #  - Por seguridad humana, por defecto SOLO se decide en la fecha límite
        #    (ventana completa). p_human se sigue calculando cada 20 ms y está
        #    disponible en current_p_human() para monitoreo/early-cut opcional.
        if self.cfg.allow_early_cut and self.elapsed_ms >= cfg.t_min_ms:
            if p <= cfg.p_buzon:                       # buzón segurísimo -> cortar ya
                return self._finish(Label.BUZON, p, feats)
            if p >= cfg.p_human_decide:                # humano segurísimo -> asesor ya
                return self._finish(Label.HUMAN, p, feats)
            # en incertidumbre seguimos escuchando hasta la fecha límite
        if self.elapsed_ms >= cfg.t_max_ms:            # fecha límite: clasificación final
            lbl = self._decide(p)
            return self._finish(lbl, p, feats, forced=True)
        return None

    def _decide(self, p: float) -> Label:
        """Mapea p_human -> destino con tus dos umbrales y la zona de incertidumbre."""
        if p <= self.cfg.p_buzon:          return Label.BUZON   # -> colgar / voicemail
        if p >= self.cfg.p_human_decide:   return Label.HUMAN   # -> asesor directo
        return Label.STT                                        # -> STT + LLM

    def current_p_human(self) -> float:
        return self._p_human(self._features()) if self.n else 0.5

    def _finish(self, label, p, feats, forced=False):
        self._done = Decision(label, p, self.elapsed_ms, forced, feats)
        return self._done

    # ---- stubs (reemplaza por tus modelos; recalibra si cambias el VAD) ----
    def _voiced(self, seg) -> bool:
        # CRÍTICO: el modelo se calibró con "voz = energía de los ÚLTIMOS 20 ms
        # > 0.01". Por defecto se usa ESO mismo, o el modelo se descalibra.
        #
        # OJO con webrtcvad: en este audio telefónico marca como "voz" ~84% de
        # los frames de un humano CALLADO (capta ruido de confort/línea), contra
        # ~12% del proxy de energía. Eso infla VAD_voiced y hace que los humanos
        # parezcan máquinas -> se pierden. Por eso NO se usa por defecto.
        # Si quieres webrtcvad: pon cfg.use_webrtcvad=True, usa agresividad 3 y
        # RECALIBRA el modelo extrayendo VAD_voiced con webrtcvad (no con energía).
        if seg.size == 0: return False
        last20 = seg[-self._samples(20):]
        if self.cfg.use_webrtcvad and _HAS_VAD:
            fr = (np.clip(last20, -1, 1) * 32767).astype(np.int16).tobytes()
            try: return bool(self._vad.is_speech(fr, self.cfg.sample_rate))
            except Exception: pass
        return float(np.sqrt(np.mean(last20 ** 2))) > self.cfg.voiced_rms_thr

    def _goertzel(self, seg) -> bool:
        if seg.size < 8: return False
        N = seg.size; k = int(0.5 + N*self.cfg.goertzel_hz/self.cfg.sample_rate)
        w = 2*math.pi*k/N; coeff = 2*math.cos(w); s1 = s2 = 0.0
        for x in seg:
            s = x + coeff*s1 - s2; s2, s1 = s1, s
        power = s2*s2 + s1*s1 - coeff*s1*s2
        total = float(np.sum(seg**2)) * N + 1e-9
        return (power/total) > self.cfg.goertzel_thr

    def _f0(self, seg):
        if seg.size < self._samples(40): return 0, 0.0
        frame, hop = self._samples(20), self._samples(10); f0s = []
        for st in range(0, seg.size-frame, hop):
            w = seg[st:st+frame]
            if np.sqrt(np.mean(w**2)) < self.cfg.rms_energy_thr: continue
            w = w - w.mean(); corr = np.correlate(w, w, mode="full")[frame-1:]
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
        print("Pon los wav en base/BASE_CONGLOMERADA/ para la demo."); raise SystemExit
    # conteo por cubo: [humanos, maquinas]
    box = {Label.BUZON: [0, 0], Label.STT: [0, 0], Label.HUMAN: [0, 0]}
    n20 = cfg.sample_rate * cfg.chunk_ms // 1000
    for f in files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        amd = StreamingAMD(cfg); dec = None
        for t in range(len(audio) // n20):
            dec = amd.feed_chunk(audio[t*n20:(t+1)*n20])
            if dec and dec.label != Label.UNDECIDED: break
        if dec is None or dec.label == Label.UNDECIDED:
            dec = Decision(amd._decide(amd.current_p_human()), amd.current_p_human(), amd.elapsed_ms, True)
        is_h = os.path.basename(f).startswith("humano")
        box[dec.label][0 if is_h else 1] += 1
    nH = sum(v[0] for v in box.values()); nM = sum(v[1] for v in box.values())
    print(f"n={len(files)}   p_buzon={cfg.p_buzon}  p_human_decide={cfg.p_human_decide}")
    print(f"{'cubo':>8}{'humanos':>9}{'máquinas':>10}   destino")
    print(f"{'BUZÓN':>8}{box[Label.BUZON][0]:>9}{box[Label.BUZON][1]:>10}   colgar/voicemail")
    print(f"{'STT':>8}{box[Label.STT][0]:>9}{box[Label.STT][1]:>10}   -> STT + LLM")
    print(f"{'HUMAN':>8}{box[Label.HUMAN][0]:>9}{box[Label.HUMAN][1]:>10}   asesor directo")
    print(f"\nHUMANOS PERDIDOS (falso buzón): {box[Label.BUZON][0]}/{nH}")
    print(f"Máquinas colgadas en el AMD:    {box[Label.BUZON][1]}/{nM}  -> el resto pasa a STT")