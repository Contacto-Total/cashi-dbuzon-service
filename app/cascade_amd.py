"""
Cascade AMD analyzer - v1

State machine de 4 etapas que compiten en paralelo entre HUMAN y BUZON:
    1. Numpy RMS         - energia/silencio
    2. Goertzel (FFT BER) - tonos puros de buzon (350/440/480/620/950/1400/1800 Hz)
    3. WebRTC VAD        - voz vs no-voz
    4. Aubio.pitch (YIN) - F0 fundamental en Hz

Cada etapa devuelve (delta_human, delta_buzon). Los deltas se acumulan en un
score corrido. Cuando un score cruza su umbral, se toma decision.

Disenado para correr sobre un ring buffer unico (~200ms). Todas las etapas
analizan el mismo acumulado.
"""
from __future__ import annotations

import math
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

# Ring buffer (ventana unica que analizan las 4 etapas)
RING_WINDOW_MS = 200

# Cada cuanto se ejecuta una pasada de analisis (no en cada chunk)
ANALYZE_EVERY_MS = 100

# Tonos buzon / SIT / dial tones tipicos
BUZON_FREQS_HZ = [350.0, 440.0, 480.0, 620.0, 950.0, 1400.0, 1800.0]
BUZON_FREQ_TOLERANCE_HZ = 30.0

# Umbrales de decision
HUMAN_THRESHOLD = 3.0
BUZON_THRESHOLD = 2.0
SCORE_CAP = 10.0  # tope duro para evitar saturacion

# WebRTC VAD
VAD_AGGRESSIVENESS = 2  # 0..3 (mas alto = mas estricto al marcar speech)
VAD_FRAME_MS = 20       # webrtcvad acepta 10/20/30 ms

# Aubio pitch
AUBIO_WIN_SIZE = 1024
AUBIO_HOP_SIZE = 512


# ── Tablas de score (delta_human, delta_buzon) ───────────────────────────────

def score_rms(rms: float) -> tuple[float, float]:
    """Energia RMS del acumulado (float32, rango 0..1)."""
    if rms < 0.005:
        return (0.0, 0.0)
    if rms < 0.01:
        return (0.1, 0.1)
    return (0.2, 0.2)


def score_goertzel(ratio: float) -> tuple[float, float]:
    """BER en bandas buzon: energia_en_bandas / energia_total."""
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
    """F0 dominante en Hz (0 = sin pitch / silencio)."""
    if hz <= 0.0 or hz < 80.0:
        return (0.0, 0.0)
    if hz < 300.0:
        return (0.7, -0.4)
    if hz < 900.0:
        return (0.1, 0.1)
    if hz < 1600.0:
        return (-0.3, 0.7)
    return (0.0, 0.2)


# ── Helpers de DSP ───────────────────────────────────────────────────────────

def band_energy_ratio(
    samples_f32: np.ndarray,
    target_freqs_hz: list[float],
    sample_rate: int,
    tol_hz: float = BUZON_FREQ_TOLERANCE_HZ,
) -> float:
    """
    Band Energy Ratio via FFT (equivalente a sumar Goertzels en cada freq).

    BER = sum(power[freq in target +/- tol]) / sum(power_total)

    Rango: 0..1. Cerca de 1 si la energia se concentra en los tonos objetivo.
    """
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


# ── Detector F0 stateful (mantiene estado del YIN entre chunks) ──────────────

class _F0Detector:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        if _AUBIO_AVAILABLE:
            self._pitch = aubio.pitch("yin", AUBIO_WIN_SIZE, AUBIO_HOP_SIZE, sample_rate)
            self._pitch.set_unit("Hz")
            self._pitch.set_silence(-40)
        else:
            self._pitch = None

    def detect_median(self, samples_f32: np.ndarray) -> float:
        """
        Corre aubio.pitch en hops consecutivos sobre el buffer y devuelve la
        mediana de F0 valida (>0 Hz). 0 si nada de pitch.
        """
        if self._pitch is None or len(samples_f32) < AUBIO_HOP_SIZE:
            return 0.0
        out = []
        for i in range(0, len(samples_f32) - AUBIO_HOP_SIZE + 1, AUBIO_HOP_SIZE):
            hop = samples_f32[i:i + AUBIO_HOP_SIZE].astype(np.float32, copy=False)
            try:
                f0 = float(self._pitch(hop)[0])
            except Exception:
                f0 = 0.0
            if f0 > 0.0:
                out.append(f0)
        if not out:
            return 0.0
        return float(np.median(out))


# ── State machine ────────────────────────────────────────────────────────────

@dataclass
class CascadeSnapshot:
    """Snapshot de la ultima pasada de analisis (para debug / telemetria)."""
    rms: float = 0.0
    goertzel_ratio: float = 0.0
    vad_active: bool = False
    f0_hz: float = 0.0
    delta_human: float = 0.0
    delta_buzon: float = 0.0
    score_human: float = 0.0
    score_buzon: float = 0.0


class CascadeAMD:
    def __init__(self, call_id: str, sample_rate: int = SAMPLE_RATE_DEFAULT):
        self.call_id = call_id
        self.sample_rate = sample_rate

        # Ring buffer en bytes (PCM L16, int16 little-endian)
        self._buffer = bytearray()
        self._window_bytes = int(sample_rate * RING_WINDOW_MS / 1000) * SAMPLE_WIDTH_BYTES
        self._analyze_every_bytes = int(sample_rate * ANALYZE_EVERY_MS / 1000) * SAMPLE_WIDTH_BYTES
        self._vad_frame_bytes = int(sample_rate * VAD_FRAME_MS / 1000) * SAMPLE_WIDTH_BYTES

        # Contadores
        self.total_bytes = 0
        self.chunks_received = 0
        self._bytes_since_last_analysis = 0

        # Estado del state machine
        self.score_human = 0.0
        self.score_buzon = 0.0
        self.last = CascadeSnapshot()

        # Detectores
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if _WEBRTC_VAD_AVAILABLE else None
        self._f0 = _F0Detector(sample_rate)

        # Decision
        self.decided = False
        self.decision_label: Optional[str] = None
        self.decision_reason: str = ""
        self.decided_at_chunk = 0
        self.decided_at_ms = 0
        self.decided_bytes = 0
        self._t0 = time.time()

    # ── API publica ──────────────────────────────────────────────────────

    def push(self, audio_bytes: bytes) -> Optional[dict]:
        """
        Recibe un chunk PCM L16. Devuelve:
          - None si todavia no toca analizar
          - dict con evento 'analysis' si se corrio una pasada (con o sin decision)
        Si la decision se tomo en esta pasada, el dict incluye 'decision'.
        """
        if not audio_bytes:
            return None

        self._buffer.extend(audio_bytes)
        self.total_bytes += len(audio_bytes)
        self.chunks_received += 1
        self._bytes_since_last_analysis += len(audio_bytes)

        # Recortar al tamano de ventana
        if len(self._buffer) > self._window_bytes:
            del self._buffer[:len(self._buffer) - self._window_bytes]

        if self.decided:
            return None

        # Aun no toca correr analisis
        if self._bytes_since_last_analysis < self._analyze_every_bytes:
            return None

        # Esperar a tener ventana completa antes de la primera pasada
        if len(self._buffer) < self._window_bytes:
            return None

        self._bytes_since_last_analysis = 0
        return self._analyze()

    def force(self) -> dict:
        """
        Forzar decision al cierre (timeout / fin de stream). Gana el score
        mas alto; empate o ambos cero -> HUMAN por defecto (politica conservadora:
        ante duda no cortar la llamada).
        """
        if not self.decided:
            if self.score_buzon > self.score_human:
                self._decide("MACHINE", "timeout-favor-buzon")
            else:
                self._decide("HUMAN", "timeout-favor-human")
        return self.build_decision_payload()

    def build_decision_payload(self) -> dict:
        denom = max(HUMAN_THRESHOLD, BUZON_THRESHOLD)
        conf = max(self.score_human, self.score_buzon) / denom if denom > 0 else 0.0
        return {
            "result": self.decision_label,
            "reason": self.decision_reason,
            "confidence": round(min(conf, 1.0), 3),
            "scores": {
                "human": round(self.score_human, 3),
                "buzon": round(self.score_buzon, 3),
            },
            "decided_at_chunk": self.decided_at_chunk,
            "decided_at_ms": self.decided_at_ms,
            "decided_bytes": self.decided_bytes,
        }

    # ── Internals ────────────────────────────────────────────────────────

    def _analyze(self) -> dict:
        samples_i16 = np.frombuffer(bytes(self._buffer), dtype=np.int16)
        samples_f32 = samples_i16.astype(np.float32) / 32768.0

        # 1) RMS
        rms = float(np.sqrt(np.mean(samples_f32 ** 2))) if samples_f32.size else 0.0
        d_h_rms, d_b_rms = score_rms(rms)

        # 2) Goertzel (BER via FFT)
        ratio = band_energy_ratio(samples_f32, BUZON_FREQS_HZ, self.sample_rate)
        d_h_goe, d_b_goe = score_goertzel(ratio)

        # 3) WebRTC VAD
        vad_active = self._run_vad(samples_i16)
        d_h_vad, d_b_vad = score_vad(vad_active)

        # 4) F0 / pitch
        f0_hz = self._f0.detect_median(samples_f32)
        d_h_f0, d_b_f0 = score_f0(f0_hz)

        delta_h = d_h_rms + d_h_goe + d_h_vad + d_h_f0
        delta_b = d_b_rms + d_b_goe + d_b_vad + d_b_f0

        self.score_human = _clamp(self.score_human + delta_h, 0.0, SCORE_CAP)
        self.score_buzon = _clamp(self.score_buzon + delta_b, 0.0, SCORE_CAP)

        self.last = CascadeSnapshot(
            rms=rms,
            goertzel_ratio=ratio,
            vad_active=vad_active,
            f0_hz=f0_hz,
            delta_human=delta_h,
            delta_buzon=delta_b,
            score_human=self.score_human,
            score_buzon=self.score_buzon,
        )

        elapsed_ms = int((time.time() - self._t0) * 1000)
        event: dict = {
            "type": "analysis",
            "n": self.chunks_received,
            "elapsed_ms": elapsed_ms,
            "stage": {
                "rms": round(rms, 4),
                "goertzel_ratio": round(ratio, 4),
                "vad": bool(vad_active),
                "f0_hz": round(f0_hz, 1),
            },
            "deltas": {
                "human": round(delta_h, 3),
                "buzon": round(delta_b, 3),
            },
            "scores": {
                "human": round(self.score_human, 3),
                "buzon": round(self.score_buzon, 3),
            },
        }

        # Decision: BUZON se evalua primero (cortar tiene prioridad sobre conectar)
        if self.score_buzon >= BUZON_THRESHOLD:
            self._decide("MACHINE", f"buzon>={BUZON_THRESHOLD}")
            event["decision"] = self.build_decision_payload()
        elif self.score_human >= HUMAN_THRESHOLD:
            self._decide("HUMAN", f"human>={HUMAN_THRESHOLD}")
            event["decision"] = self.build_decision_payload()

        return event

    def _run_vad(self, samples_i16: np.ndarray) -> bool:
        if self._vad is None:
            return False
        audio_bytes = samples_i16.tobytes()
        frame_bytes = self._vad_frame_bytes
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
        # Mayoria simple
        return n_speech * 2 >= n_frames

    def _decide(self, label: str, reason: str):
        self.decided = True
        self.decision_label = label
        self.decision_reason = reason
        self.decided_at_chunk = self.chunks_received
        self.decided_at_ms = int((time.time() - self._t0) * 1000)
        self.decided_bytes = self.total_bytes


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
