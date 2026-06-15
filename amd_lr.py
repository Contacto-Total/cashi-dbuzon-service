
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


# --- Modelos (orden de features: [RMS_energy, RMS_hi, VAD_voiced, GOE_tone, F0_std_hi]) ---
# Precisión COMPLETA, idéntica a los JSON (la fuente de verdad). No redondear.
# MODEL_2000 == amd_final_model.json   |   MODEL_2800 == amd_final_model_2800.json
MODEL_2000 = {
    "mu":   [0.4568312434691746, 0.2301462904911182, 0.37367206548241033,
             0.051506443747823226, 0.2444542091382707],
    "sd":   [0.26177556001144264, 0.238938962714524, 0.23550205437603539,
             0.03483800862486579, 0.27299164590499875],
    "coef": [-1.5309755822130389, 1.2807213149613483, -3.194310863299281,
             -0.9442103123734442, 1.7897188490047853],
    "b0":   -0.9178130745334432,
}
MODEL_2800 = {
    "mu":   [0.47219982634581953, 0.21658707516745257, 0.3812174398412308,
             0.06574051103944431, 0.24290663011559524],
    "sd":   [0.2203850430547032, 0.22146361392021852, 0.19723553124754079,
             0.05178931485837344, 0.25764758432107693],
    "coef": [-1.8585155006234457, 1.8100166535340352, -2.72050472954208,
             -1.8286361444624832, 1.7564124904235963],
    "b0":   -1.4373634571680922,
}
_ORDER = ["RMS_energy", "RMS_hi", "VAD_voiced", "GOE_tone", "F0_std_hi"]


@dataclass
class AMDConfig:
    sample_rate: int = 8000
    chunk_ms: int = 20
    win_rms_ms: int = 40
    win_goertzel_ms: int = 60
    win_vad_ms: int = 80
    win_f0_ms: int = 100

    # Cortes de banda por tick (de los datos)
    voiced_rms_thr: float = 0.01      # VAD por energía (con lo que se ENTRENÓ)
    use_webrtcvad: bool = False       # True solo si recalibras con webrtcvad
    webrtcvad_aggressiveness: int = 3
    rms_energy_thr: float = 0.005
    rms_hi_thr: float = 0.03
    goertzel_hz: float = 440.0
    goertzel_thr: float = 0.05
    f0_std_hi: float = 40.0
    f0_min_hz: float = 80.0
    f0_max_hz: float = 300.0
    f0_n_min: int = 3

    # --- Decisión de DOS ETAPAS ---
    t_min_ms: int = 800               # no cortar buzón antes (early = poco fiable)
    t_stage1_ms: int = 2000           # 1ra decisión (MODEL_2000)
    t_stage2_ms: int = 2800           # desempate de los dudosos (MODEL_2800)

    # Umbrales (tus dos niveles de confianza), iguales en ambas etapas:
    #   p_buzon = 0.0276 -> justo bajo el peor humano (0 humanos perdidos).
    #   p_human_decide = 0.80 -> 80% de humanos al asesor directo; 2 máquinas se cuelan.
    p_buzon: float = 0.0276
    p_human_decide: float = 0.80

    # Early-cut OFF por defecto: decidir con la ventana completa de cada etapa es
    # lo más seguro para el humano (un humano puede tener un bajón transitorio).
    allow_early_cut: bool = False


class Label(Enum):
    HUMAN = "HUMAN"      # p_human >= p_human_decide -> ASESOR DIRECTO (se salta GPT)
    STT   = "STT"        # duda tras la 2da etapa    -> STT + LLM
    BUZON = "BUZON"      # p_human <= p_buzon         -> colgar / dejar mensaje
    UNDECIDED = "UNDECIDED"


@dataclass
class Decision:
    label: Label
    p_human: float
    elapsed_ms: int
    forced: bool = False        # resuelto en una fecha límite (deadline de etapa)
    stage: int = 0              # 1 = decidió a 2000ms, 2 = decidió a 2800ms
    features: Optional[dict] = None


class StreamingAMD:
    def __init__(self, cfg: Optional[AMDConfig] = None):
        self.cfg = cfg or AMDConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self.elapsed_ms = 0
        self._done: Optional[Decision] = None
        self._stage1_done = False           # ya pasó (sin resolver) la etapa 1
        self._last_p = 0.5
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

    def _p_human(self, f, model) -> float:
        z = model["b0"]
        for i, k in enumerate(_ORDER):
            z += model["coef"][i] * (f[k] - model["mu"][i]) / model["sd"][i]
        return self._sigmoid(z)

    def _decide(self, p: float) -> Label:
        if p <= self.cfg.p_buzon:          return Label.BUZON
        if p >= self.cfg.p_human_decide:   return Label.HUMAN
        return Label.STT

    def feed_chunk(self, chunk: np.ndarray) -> Optional[Decision]:
        """Se llama CADA 20 ms. Recalcula p_human y, en las fechas límite de
        cada etapa, decide. Devuelve Decision cuando ya hay veredicto, o None."""
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

        # (3)+(4) fracciones
        feats = self._features()

        # --- Early-cut opcional (solo etapa 1, OFF por defecto) ---
        if (cfg.allow_early_cut and not self._stage1_done
                and self.elapsed_ms >= cfg.t_min_ms):
            p = self._p_human(feats, MODEL_2000); self._last_p = p
            if p <= cfg.p_buzon:        return self._finish(Label.BUZON, p, feats, stage=1)
            if p >= cfg.p_human_decide: return self._finish(Label.HUMAN, p, feats, stage=1)

        # --- ETAPA 1: fecha límite a 2000 ms ---
        if not self._stage1_done and self.elapsed_ms >= cfg.t_stage1_ms:
            self._stage1_done = True
            p = self._p_human(feats, MODEL_2000); self._last_p = p
            if p <= cfg.p_buzon:        return self._finish(Label.BUZON, p, feats, stage=1)
            if p >= cfg.p_human_decide: return self._finish(Label.HUMAN, p, feats, stage=1)
            # en duda -> NO va a GPT; sigue escuchando hasta la etapa 2
            return None

        # --- ETAPA 2: desempate a 2800 ms (solo dudosos) ---
        if self._stage1_done and self.elapsed_ms >= cfg.t_stage2_ms:
            p = self._p_human(feats, MODEL_2800); self._last_p = p
            return self._finish(self._decide(p), p, feats, stage=2, forced=True)

        # monitoreo: p con el modelo de la etapa en curso
        self._last_p = self._p_human(feats, MODEL_2000 if not self._stage1_done else MODEL_2800)
        return None

    def force_decision(self) -> Decision:
        """Para cuando la llamada se corta antes de la fecha límite (best-effort):
        decide con el modelo de la etapa alcanzada."""
        if self._done is not None:
            return self._done
        feats = self._features()
        model = MODEL_2800 if self.elapsed_ms >= self.cfg.t_stage2_ms else MODEL_2000
        p = self._p_human(feats, model) if self.n else 0.5
        stage = 2 if model is MODEL_2800 else 1
        return self._finish(self._decide(p), p, feats, stage=stage, forced=True)

    def current_p_human(self) -> float:
        return self._last_p if self.n else 0.5

    def _finish(self, label, p, feats, stage=0, forced=False):
        self._done = Decision(label, p, self.elapsed_ms, forced, stage, feats)
        return self._done

    # ---- stubs (reemplaza por tus modelos; recalibra si cambias el VAD) ----
    def _voiced(self, seg) -> bool:
        # CRÍTICO: el modelo se calibró con "voz = energía de los ÚLTIMOS 20 ms
        # > 0.01". webrtcvad infla VAD_voiced en humanos callados y los pierde;
        # por eso está OFF por defecto. Si lo activas, RECALIBRA ambos modelos.
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
    box = {Label.BUZON: [0, 0], Label.STT: [0, 0], Label.HUMAN: [0, 0]}
    stage_count = {1: 0, 2: 0}
    n20 = cfg.sample_rate * cfg.chunk_ms // 1000
    for f in files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        amd = StreamingAMD(cfg); dec = None
        for t in range(len(audio) // n20):
            dec = amd.feed_chunk(audio[t*n20:(t+1)*n20])
            if dec and dec.label != Label.UNDECIDED: break
        if dec is None or dec.label == Label.UNDECIDED:
            dec = amd.force_decision()
        is_h = os.path.basename(f).startswith("humano")
        box[dec.label][0 if is_h else 1] += 1
        if dec.stage in stage_count: stage_count[dec.stage] += 1
    nH = sum(v[0] for v in box.values()); nM = sum(v[1] for v in box.values())
    print(f"n={len(files)}  p_buzon={cfg.p_buzon}  p_human_decide={cfg.p_human_decide}"
          f"  etapas={cfg.t_stage1_ms}/{cfg.t_stage2_ms}ms")
    print(f"{'cubo':>8}{'humanos':>9}{'máquinas':>10}   destino")
    print(f"{'BUZÓN':>8}{box[Label.BUZON][0]:>9}{box[Label.BUZON][1]:>10}   colgar/voicemail")
    print(f"{'STT':>8}{box[Label.STT][0]:>9}{box[Label.STT][1]:>10}   -> GPT")
    print(f"{'HUMAN':>8}{box[Label.HUMAN][0]:>9}{box[Label.HUMAN][1]:>10}   asesor directo")
    print(f"\nDecididos en etapa 1 (2000ms): {stage_count[1]}   |   en etapa 2 (2800ms): {stage_count[2]}")
    print(f"HUMANOS PERDIDOS (falso buzón): {box[Label.BUZON][0]}/{nH}")
    print(f"A GPT: {box[Label.STT][0]+box[Label.STT][1]}/{nH+nM}")