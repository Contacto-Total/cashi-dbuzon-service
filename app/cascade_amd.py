"""
Cascade AMD analyzer - v2

Diseno actual (acordado en sesion):
- Acumulador (no ring rotativo): PCM L16 crece de 0 hasta 1500ms y se detiene.
- En CADA chunk recibido (~50/s a 20ms) corren las 4 etapas sobre TODO el acumulado.
- Cada chunk emite un solo evento type='analysis' con resultado de los 4 modelos,
  sus deltas individuales, los scores acumulados y la contribucion por modelo.
- Cuando un score cruza umbral -> decision (incluye triggered_by + last_delta_from).
- Si llega al cap 1500ms sin decision -> whisper_needed=True (main.py corre Whisper).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import webrtcvad
    _WEBRTC_VAD_AVAILABLE = True
except ImportError:
    _WEBRTC_VAD_AVAILABLE = False

try:
    import aubio
    _AUBIO_AVAILABLE = True
except ImportError:
    _AUBIO_AVAILABLE = False


# ── Configuracion ────────────────────────────────────────────────────────────

SAMPLE_RATE_DEFAULT = 8000
SAMPLE_WIDTH_BYTES = 2  # PCM L16

# Cap del acumulador antes de pasar a Whisper (Limite de analisis de modelos anteriores)
MAX_ACCUMULATOR_MS = 1500

# Tonos buzon / SIT / dial-tones tipicos
BUZON_FREQS_HZ = [350.0, 440.0, 480.0, 620.0, 950.0, 1400.0, 1800.0]
BUZON_FREQ_TOLERANCE_HZ = 30.0

# Umbrales del state machine
HUMAN_THRESHOLD = 3.0
BUZON_THRESHOLD = 2.0
SCORE_CAP = 10.0 #limite superior para los scores (evita que un score muy alto)

# WebRTC VAD
VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 20

# Aubio pitch (YIN)
AUBIO_WIN_SIZE = 1024
AUBIO_HOP_SIZE = 512
F0_HISTORY_MAX = 30  # mantener mediana sobre ultimos 30 hops validos


# ── Tablas de score (delta_human, delta_buzon) ───────────────────────────────

def score_rms(rms: float) -> tuple[float, float]:
    if rms < 0.005:
        return (0.0, 0.0)
    if rms < 0.01:
        return (0.1, 0.1)
    return (0.2, 0.2)


def score_goertzel(ratio: float) -> tuple[float, float]:
    if ratio < 0.3:
        return (0.0, 0.0)
    if ratio < 0.5:
        return (-0.1, 0.3)
    if ratio < 0.7:
        return (-0.2, 0.6)
    return (-0.3, 1.0)


def score_vad(is_speech: bool) -> tuple[float, float]:
    if is_speech:
        return (0.5, -0.3)
    return (0.0, 0.1)


def score_f0(hz: float) -> tuple[float, float]:
    if hz <= 0.0 or hz < 80.0:
        return (0.0, 0.0)
    if hz < 300.0:
        return (0.7, -0.4)
    if hz < 900.0:
        return (0.1, 0.1)
    if hz < 1600.0:
        return (-0.3, 0.7)
    return (0.0, 0.2)


# ── DSP helpers ──────────────────────────────────────────────────────────────

def band_energy_ratio(
    samples_f32: np.ndarray,
    target_freqs_hz: list[float],
    sample_rate: int,
    tol_hz: float = BUZON_FREQ_TOLERANCE_HZ,
) -> float:
    """BER via FFT: sum(power en bandas objetivo) / sum(power total). Rango 0..1."""
    n = len(samples_f32)
    if n == 0:
        return 0.0
    spec = np.fft.rfft(samples_f32)
    power = (spec.real ** 2 + spec.imag ** 2)
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band_total = 0.0
    for f in target_freqs_hz:
        mask = np.abs(freqs - f) <= tol_hz
        band_total += float(power[mask].sum())
    return band_total / total


class _F0Detector:
    """
    Aubio.pitch incremental: procesa solo los hops nuevos del acumulador y
    devuelve la mediana de los ultimos F0 validos.
    """
    def __init__(self, sample_rate: int):
        self._sample_rate = sample_rate
        self._processed = 0
        self._history: list[float] = []
        if _AUBIO_AVAILABLE:
            self._pitch = aubio.pitch("yin", AUBIO_WIN_SIZE, AUBIO_HOP_SIZE, sample_rate)
            self._pitch.set_unit("Hz")
            self._pitch.set_silence(-40)
        else:
            self._pitch = None

    def update(self, accumulator_f32: np.ndarray) -> float:
        if self._pitch is None:
            return 0.0
        i = self._processed
        while i + AUBIO_HOP_SIZE <= len(accumulator_f32):
            hop = accumulator_f32[i:i + AUBIO_HOP_SIZE].astype(np.float32, copy=False)
            try:
                f0 = float(self._pitch(hop)[0])
            except Exception:
                f0 = 0.0
            if f0 > 0.0:
                self._history.append(f0)
                if len(self._history) > F0_HISTORY_MAX:
                    self._history.pop(0)
            i += AUBIO_HOP_SIZE
        self._processed = i
        if not self._history:
            return 0.0
        return float(np.median(self._history))


# ── State machine ────────────────────────────────────────────────────────────

class CascadeAMD:
    MODELS = ("rms", "goertzel", "vad", "f0")

    def __init__(self, call_id: str, sample_rate: int = SAMPLE_RATE_DEFAULT):
        self.call_id = call_id
        self.sample_rate = sample_rate

        # Acumulador PCM L16
        self._buffer = bytearray()
        self._max_bytes = int(sample_rate * MAX_ACCUMULATOR_MS / 1000) * SAMPLE_WIDTH_BYTES
        self._vad_frame_bytes = int(sample_rate * VAD_FRAME_MS / 1000) * SAMPLE_WIDTH_BYTES

        # Contadores
        self.total_bytes = 0
        self.chunks_received = 0

        # State machine
        self.score_human = 0.0
        self.score_buzon = 0.0
        self.contrib_human = {m: 0.0 for m in self.MODELS}
        self.contrib_buzon = {m: 0.0 for m in self.MODELS}
        self.last_delta_from: Optional[str] = None

        # Detectores
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if _WEBRTC_VAD_AVAILABLE else None
        self._f0 = _F0Detector(sample_rate)

        # Decision
        self.decided = False
        self.whisper_needed = False
        self.decision_label: Optional[str] = None
        self.decision_reason: str = ""
        self.decided_at_chunk = 0
        self.decided_at_ms = 0
        self.decided_bytes = 0
        self._t0 = time.time()

    # ── API publica ──────────────────────────────────────────────────────

    def get_accumulator_f32(self) -> np.ndarray:
        """Audio acumulado como float32 mono (para pasar a Whisper)."""
        samples_i16 = np.frombuffer(bytes(self._buffer), dtype=np.int16)
        return samples_i16.astype(np.float32) / 32768.0

    def push(self, audio_bytes: bytes) -> Optional[dict]:
        """
        Recibe un chunk PCM L16. Corre las 4 etapas sobre el acumulado.
        Devuelve evento type='analysis' (o None si ya hubo decision / whisper).
        El evento puede incluir 'decision' inline o 'whisper_needed': True.
        """
        if not audio_bytes or self.decided or self.whisper_needed:
            return None

        # Anexar al acumulador respetando el cap
        space_left = self._max_bytes - len(self._buffer)
        if space_left <= 0:
            self.whisper_needed = True
            return {
                "type": "whisper_pending",
                "n": self.chunks_received,
                "elapsed_ms": int((time.time() - self._t0) * 1000),
            }

        to_add = audio_bytes[:space_left]
        self._buffer.extend(to_add)
        self.total_bytes += len(audio_bytes)
        self.chunks_received += 1

        event = self._analyze()

        # Llegamos al cap en este push y no se decidio -> marcar Whisper
        if not self.decided and len(self._buffer) >= self._max_bytes:
            self.whisper_needed = True
            event["whisper_needed"] = True

        return event

    def force(self) -> dict:
        """Forzar decision al cierre sin haber cruzado umbral. HUMAN por defecto."""
        if not self.decided:
            if self.score_buzon > self.score_human:
                self.last_delta_from = self._top_model(self.contrib_buzon)
                self._decide("MACHINE", "timeout-favor-buzon")
                return self._build_decision_payload(side="buzon")
            self.last_delta_from = self._top_model(self.contrib_human)
            self._decide("HUMAN", "timeout-favor-human")
            return self._build_decision_payload(side="human")
        side = "buzon" if self.decision_label == "MACHINE" else "human"
        return self._build_decision_payload(side=side)

    # ── Internals ────────────────────────────────────────────────────────

    def _analyze(self) -> dict:
        samples_i16 = np.frombuffer(bytes(self._buffer), dtype=np.int16)
        samples_f32 = samples_i16.astype(np.float32) / 32768.0
        n_samples = len(samples_f32)

        # 1) RMS
        rms = float(np.sqrt(np.mean(samples_f32 ** 2))) if n_samples else 0.0
        dh_rms, db_rms = score_rms(rms)

        # 2) Goertzel (BER via FFT)
        ratio = band_energy_ratio(samples_f32, BUZON_FREQS_HZ, self.sample_rate)
        dh_goe, db_goe = score_goertzel(ratio)

        # 3) WebRTC VAD
        vad_active = self._run_vad(samples_i16)
        dh_vad, db_vad = score_vad(vad_active)

        # 4) F0 / Pitch
        f0_hz = self._f0.update(samples_f32)
        dh_f0, db_f0 = score_f0(f0_hz)

        # Acumular contribuciones por modelo y scores totales
        self.contrib_human["rms"] += dh_rms
        self.contrib_human["goertzel"] += dh_goe
        self.contrib_human["vad"] += dh_vad
        self.contrib_human["f0"] += dh_f0
        self.contrib_buzon["rms"] += db_rms
        self.contrib_buzon["goertzel"] += db_goe
        self.contrib_buzon["vad"] += db_vad
        self.contrib_buzon["f0"] += db_f0

        delta_h_total = dh_rms*(0.15) + dh_goe*(0.30) + dh_vad*(0.25) + dh_f0*(0.30)
        delta_b_total = db_rms*(0.15) + db_goe*(0.30) + db_vad*(0.25) + db_f0*(0.30)

        self.score_human = _clamp(self.score_human + delta_h_total, 0.0, SCORE_CAP)
        self.score_buzon = _clamp(self.score_buzon + delta_b_total, 0.0, SCORE_CAP)

        elapsed_ms = int((time.time() - self._t0) * 1000)
        accumulator_ms = int(n_samples * 1000 / self.sample_rate)

        event: dict = {
            "type": "analysis",
            "n": self.chunks_received,
            "elapsed_ms": elapsed_ms,
            "accumulator_ms": accumulator_ms,
            "models": {
                "rms":      {"value": round(rms, 4),    "dh": round(dh_rms, 3), "db": round(db_rms, 3)},
                "goertzel": {"value": round(ratio, 4),  "dh": round(dh_goe, 3), "db": round(db_goe, 3)},
                "vad":      {"value": bool(vad_active), "dh": round(dh_vad, 3), "db": round(db_vad, 3)},
                "f0":       {"value": round(f0_hz, 1),  "dh": round(dh_f0, 3),  "db": round(db_f0, 3)},
            },
            "deltas": {"human": round(delta_h_total, 3), "buzon": round(delta_b_total, 3)},
            "scores": {"human": round(self.score_human, 3), "buzon": round(self.score_buzon, 3)},
            "contrib_human": {k: round(v, 3) for k, v in self.contrib_human.items()},
            "contrib_buzon": {k: round(v, 3) for k, v in self.contrib_buzon.items()},
        }

        # Check umbrales (BUZON primero, cortar tiene prioridad sobre conectar)
        per_model_dh = {"rms": dh_rms, "goertzel": dh_goe, "vad": dh_vad, "f0": dh_f0}
        per_model_db = {"rms": db_rms, "goertzel": db_goe, "vad": db_vad, "f0": db_f0}

        if self.score_buzon >= BUZON_THRESHOLD:
            self.last_delta_from = max(per_model_db, key=lambda k: per_model_db[k])
            self._decide("MACHINE", f"buzon>={BUZON_THRESHOLD}")
            event["decision"] = self._build_decision_payload(side="buzon")
        elif self.score_human >= HUMAN_THRESHOLD:
            self.last_delta_from = max(per_model_dh, key=lambda k: per_model_dh[k])
            self._decide("HUMAN", f"human>={HUMAN_THRESHOLD}")
            event["decision"] = self._build_decision_payload(side="human")

        return event

    def _run_vad(self, samples_i16: np.ndarray) -> bool:
        if self._vad is None:
            return False
        audio_bytes = samples_i16.tobytes()
        frame_bytes = self._vad_frame_bytes
        if frame_bytes <= 0 or len(audio_bytes) < frame_bytes:
            return False
        n_speech = 0
        n_frames = 0
        for i in range(0, len(audio_bytes) - frame_bytes + 1, frame_bytes):
            frame = audio_bytes[i:i + frame_bytes]
            try:
                if self._vad.is_speech(frame, self.sample_rate):
                    n_speech += 1
            except Exception:
                pass
            n_frames += 1
        if n_frames == 0:
            return False
        return n_speech * 2 >= n_frames

    def _decide(self, label: str, reason: str):
        self.decided = True
        self.decision_label = label
        self.decision_reason = reason
        self.decided_at_chunk = self.chunks_received
        self.decided_at_ms = int((time.time() - self._t0) * 1000)
        self.decided_bytes = self.total_bytes

    def _build_decision_payload(self, side: str) -> dict:
        denom = HUMAN_THRESHOLD if side == "human" else BUZON_THRESHOLD
        score_won = self.score_human if side == "human" else self.score_buzon
        conf = min(score_won / denom, 1.0) if denom > 0 else 0.0

        contrib_side = self.contrib_human if side == "human" else self.contrib_buzon
        triggered_by = self._top_model(contrib_side)

        return {
            "result": self.decision_label,
            "reason": self.decision_reason,
            "confidence": round(conf, 3),
            "scores": {
                "human": round(self.score_human, 3),
                "buzon": round(self.score_buzon, 3),
            },
            "contrib_human": {k: round(v, 3) for k, v in self.contrib_human.items()},
            "contrib_buzon": {k: round(v, 3) for k, v in self.contrib_buzon.items()},
            "triggered_by": triggered_by,
            "last_delta_from": self.last_delta_from,
            "decided_at_chunk": self.decided_at_chunk,
            "decided_at_ms": self.decided_at_ms,
            "decided_bytes": self.decided_bytes,
        }

    @staticmethod
    def _top_model(contrib: dict) -> str:
        return max(contrib, key=lambda k: contrib[k])

    def set_whisper_decision(self, result: str, reason: str, transcription: str = "",
                              confidence: float = 0.0) -> dict:
        """
        Llamado por main.py despues de correr Whisper sobre el acumulado.
        Cierra el state machine con la decision de Whisper.
        """
        self.decided = True
        self.decision_label = result
        self.decision_reason = f"whisper:{reason}"
        self.decided_at_chunk = self.chunks_received
        self.decided_at_ms = int((time.time() - self._t0) * 1000)
        self.decided_bytes = self.total_bytes
        self.last_delta_from = "whisper"
        side = "buzon" if result == "MACHINE" else "human"

        payload = self._build_decision_payload(side=side)
        payload["confidence"] = round(confidence, 3)
        payload["transcription"] = transcription
        payload["triggered_by"] = "whisper"
        return payload


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
