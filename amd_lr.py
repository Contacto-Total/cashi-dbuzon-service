"""
amd_lr.py — Detección de contestadora (AMD) en streaming con EMBUDO PROGRESIVO.

Idea central (validada sobre 2687 audios reales de producción):
  - El audio entra en chunks de 20 ms y se va acumulando.
  - En cada checkpoint (600, 800, ... 2000 ms) se calculan 8 features sobre la
    ventana acumulada y un modelo logístico da p_human.
  - Regla SUAVIZADA: p_efectiva = (p_actual + p_anterior)/2, y para colgar se
    exigen >=2 checkpoints (nunca se cuelga en el primero). Así un humano que
    baja un instante no se pierde.
  - Lo que sigue incierto a los 2000 ms -> se manda a STT/LLM (ChatGPT).

QUÉ SEPARA REALMENTE A LAS CLASES (importante, no es lo intuitivo):
  Etiqueta "humano" = una PERSONA contestó, hable o no. En la muestra real:
  42% de los humanos son silencio casi total, 45% ruido/energía baja, y solo
  11% tiene voz detectable. Las máquinas, en cambio, hablan en 97% de los casos.
  O sea: el discriminante NO es "el humano habla", es "la grabación siempre
  habla y la persona puede quedarse callada". Por eso f0n_rate (voz sostenida
  con F0 estable) domina con coeficiente NEGATIVO en los 8 checkpoints.
  Consecuencia: el caso de riesgo es el humano que contesta hablando de
  corrido; está poco representado (~11%) y ahí se concentra la pérdida.

8 features (de varias librerías):
  f0n_rate, f0std_hi  -> F0 (autocorrelación; en prod usa tu detector aubio)
  g620                -> Goertzel 620 Hz
  mfcc1, mfcc3, mfcc6 -> librosa MFCC
  rolloff             -> librosa spectral rolloff
  sil_slope           -> silero-vad: voz(2da mitad) - voz(1ra mitad)

CALIBRACIÓN (funnel_model.json v5): 2687 audios reales de producción etiquetados
a mano uno por uno (371 human, 2316 nonhuman). Etiquetas = ground truth.
Arquitectura, features, regla de decisión y C=1.0: SIN CAMBIOS.

v5 = v3 + UN SOLO cambio: el piso del clip de gates_human sube de 0.6 a 0.9.
  gates_human = clip(q0.997(máquinas OOF) + 0.02, 0.9, 0.995)
  Solo se mueven 3 de 8 valores: 600ms 0.8767->0.9, 1800ms 0.814->0.9,
  2000ms 0.6->0.9. gates_buzon, coeficientes, mu y sd quedan idénticos al v3.
  Motivo (medido): 10 de las 21 fugas (48%) ocurrían a 1800ms, donde el piso 0.6
  dejaba la compuerta en 0.814 y el modelo entrena con solo 27 humanos (16 a
  2000ms). Son los checkpoints de modelo más débil y el piso los volvía laxos.

MEDIDO sobre 10 splits 75/25 independientes (mismas semillas para ambos):
                     FN humano->BUZÓN   FP máq->HUMANO   máq cortadas   STT
    v3 (base)          3.70% ± 1.11         0.36%            70.4%      35.7%
    v5 (activo)        3.70% ± 1.11         0.21%            70.4%      36.2%
  FN y corte idénticos por construcción (el piso de gates_human no toca la ruta
  de BUZÓN). La ganancia es toda en falsos positivos: -42%. Costo: 2.9% de los
  humanos pasan de HUMANO-directo a STT, que es camino seguro.
  Otro punto de la curva: copia gates_alternativas["qBx"] sobre gates_buzon
  (qB0.005 -> FN 3.37% / 62.6% cortadas). La pérdida NO baja a cero: hay
  solapamiento real entre las clases.

PROBADO Y DESCARTADO POR NO MEJORAR (no reintentar sin data nueva):
  - gates_human = q0.98 (era el v4): bajaba el corte 70%->44% y multiplicaba la
    fuga x10. Su justificación —que 180 humanos llegaban mudos al STT y
    terminaban en buzón por `if not text: return buzon`— es FALSA:
    transcribe_whi_gtp4_mini normaliza a pico 0.95 (ganancia mediana x99) y el
    84.4% de esos clips SÍ tiene contenido tras normalizar; ninguno es silencio
    digital. Medir el clip crudo con el T_sil del gate fue el error.
  - Regularización C: 0.1 y 0.3 empeoran FN +0.76 pts; 3.0 gana 0.11 (ruido) y
    pierde 1.8 pts de corte. Se mantiene C=1.0.
  - Calendarios de gates_buzon por checkpoint: los 4 probados empeoran FN (+0.65
    a +1.52). Endurecer las compuertas tempranas NO salva humanos: sobreviven a
    800/1000ms y los mata la compuerta de 1200+. Se mantiene qB0.01 uniforme.
  - Compuerta atípica de 1400ms (0.038): causa 1 pérdida de 34 en 10 splits.
    No es el problema; no se toca.

DÓNDE SE PIERDEN (diagnóstico, 10 splits): 20 de 34 pérdidas ocurren a 800 y
1000ms, que es donde también se corta el 78% de las máquinas. Ese es el cuello
de botella real; moverlo requiere data nueva, no reajustar umbrales.

silero_vad.onnx debe estar junto a este archivo (o ajusta la ruta en AMDConfig).
"""
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


_SILERO_SESS = None
def _get_silero_sess(path):
    """Carga la sesión ONNX de silero UNA sola vez (compartida entre llamadas)."""
    global _SILERO_SESS
    if _SILERO_SESS is None and _HAS_ORT and os.path.exists(path):
        _ort.set_default_logger_severity(3)
        so = _ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        _SILERO_SESS = _ort.InferenceSession(path, so)
    return _SILERO_SESS


class _SileroVAD:
    FRAME = 256

    def __init__(self, path: str):
        self.sess = _get_silero_sess(path)   # cacheado: 1 sola carga, no por llamada
        self.ok = self.sess is not None
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


_FUNNEL_CACHE = {}
def _get_funnel_model(path):
    """Carga/parsea el funnel_model.json UNA sola vez por ruta (cacheado)."""
    if path not in _FUNNEL_CACHE:
        _FUNNEL_CACHE[path] = _FunnelModel(path)
    return _FUNNEL_CACHE[path]


class StreamingAMD:
    def __init__(self, cfg: Optional[AMDConfig] = None):
        self.cfg = cfg or AMDConfig()
        self.model = _get_funnel_model(self.cfg.model_path)
        self.checkpoints = self.model.checkpoints
        self._silero = _SileroVAD(self.cfg.silero_path) if self.cfg.use_silero else None
        self.using_silero = bool(self._silero and self._silero.ok)
        self.reset()

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_cp = 0
        self._prev_p = None      # p crudo del checkpoint anterior (para suavizado)
        self._last_pe = 0.5      # p efectiva más reciente (para reporte)
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
            return Decision(Outcome.UNDECIDED, self._last_pe, self.elapsed_ms)
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
        return Decision(Outcome.UNDECIDED, self._last_pe, self.elapsed_ms)

    def force_decision(self) -> Decision:
        """El audio terminó (o el integrador exige respuesta) sin decisión del
        embudo. NUNCA apostamos BUZÓN/HUMANO con ventana parcial: lo seguro es
        derivar a STT con el audio disponible."""
        if self._done:
            return Decision(Outcome.UNDECIDED, self._last_pe, self.elapsed_ms)
        self._done = True
        return Decision(Outcome.SEND_TO_STT, self._last_pe, self.elapsed_ms,
                        f"audio termino sin decision (pe={self._last_pe:.3f}); a STT")

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
        """Regla SUAVIZADA (v2): p_efectiva = promedio de p actual y p del
        checkpoint anterior. BUZÓN requiere al menos 2 checkpoints (nunca se
        cuelga en el primero); un bajón momentáneo no mata a un humano."""
        feat = self._feature_vector(ci)
        p = self.model.p_human(ci, feat)
        pe = p if self._prev_p is None else 0.5 * (p + self._prev_p)
        self._last_pe = pe
        cp = self.checkpoints[ci]
        gB = self.model.gates_buzon[ci]; gH = self.model.gates_human[ci]
        out, reason = Outcome.UNDECIDED, ""
        if pe >= gH:
            out, reason = Outcome.HUMANO, f"pe={pe:.2f}>=H({gH})"
        elif self._prev_p is not None and pe <= gB:
            out, reason = Outcome.BUZON, f"pe={pe:.4f}<=B({gB}) con 2 checkpoints"
        if out is Outcome.UNDECIDED and (ci == len(self.checkpoints) - 1 or force):
            if pe >= 0.5:
                reason = f"STT: tiende a humano (pe={pe:.2f})"
            elif pe <= gB * 3:
                reason = f"STT: maquina borderline (pe={pe:.4f})"
            else:
                reason = f"STT: zona media (pe={pe:.2f})"
            out = Outcome.SEND_TO_STT
        self._prev_p = p
        return Decision(out, pe, cp, reason)


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