"""
AMD Detector - Deteccion de Buzon de Voz
Usa Silero VAD (ONNX, sin PyTorch) + Resemblyzer + SVM
"""
import logging
import os
import urllib.request
import numpy as np
import scipy.signal
import onnxruntime as ort

from app.classifier import VoiceClassifier
from app.config import (
    VAD_SAMPLE_RATE,
    VAD_CHUNK_SAMPLES,
    VAD_THRESHOLD,
    AUDIO_INPUT_SAMPLE_RATE,
    AMD_MIN_VOICE_SECONDS,
    AMD_FALLBACK_SECONDS,
    AMD_CONFIDENCE_THRESHOLD,
    BEEP_FREQ_MIN,
    BEEP_FREQ_MAX,
    BEEP_ENERGY_THRESHOLD,
    BEEP_MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# URL oficial del modelo ONNX de Silero VAD
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SILERO_ONNX_PATH = "models/silero_vad.onnx"


def _download_silero_onnx():
    """Descarga el modelo ONNX de Silero VAD si no existe."""
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(SILERO_ONNX_PATH):
        logger.info(f"Descargando Silero VAD ONNX desde {SILERO_ONNX_URL}...")
        urllib.request.urlretrieve(SILERO_ONNX_URL, SILERO_ONNX_PATH)
        logger.info(f"Silero VAD ONNX guardado en {SILERO_ONNX_PATH}")
    else:
        logger.info(f"Silero VAD ONNX ya existe en {SILERO_ONNX_PATH}")


class SileroVAD:
    """
    Wrapper liviano para Silero VAD usando ONNX Runtime.
    Sin PyTorch. Pesa ~2MB el modelo, ~50MB onnxruntime.
    """

    def __init__(self, model_path: str = SILERO_ONNX_PATH):
        _download_silero_onnx()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # Silenciar logs de ONNX
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self._reset_state()
        logger.info("Silero VAD ONNX inicializado.")

    def _reset_state(self):
        """Resetea el estado interno del modelo (necesario por sesion de llamada)."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, audio_chunk: np.ndarray, sample_rate: int = VAD_SAMPLE_RATE) -> float:
        """
        Procesa un chunk de audio y retorna probabilidad de voz.

        Args:
            audio_chunk: float32 array de VAD_CHUNK_SAMPLES (512) samples a 16kHz
            sample_rate: debe ser 16000

        Returns:
            float: probabilidad de voz (0.0 - 1.0)
        """
        # Silero ONNX espera shape (1, chunk_size)
        x = audio_chunk.reshape(1, -1).astype(np.float32)
        sr = np.array(sample_rate, dtype=np.int64)

        out, self._h, self._c = self.session.run(
            None,
            {"input": x, "sr": sr, "h": self._h, "c": self._c}
        )
        return float(out[0][0])

    def clone_state(self):
        """Retorna copia del estado actual (para sesiones paralelas)."""
        return np.copy(self._h), np.copy(self._c)

    def set_state(self, h, c):
        """Restaura estado (para sesiones paralelas)."""
        self._h = h
        self._c = c


class AMDDetector:
    """
    Detector principal AMD. Singleton compartido entre todas las sesiones.
    Contiene: Silero VAD ONNX y clasificador Resemblyzer+SVM.
    """

    def __init__(self):
        logger.info("Cargando Silero VAD (ONNX, sin PyTorch)...")
        # El modelo ONNX base es compartido (stateless en el grafo)
        # El estado (h, c) se maneja por sesion en AMDSession
        self._vad_base = SileroVAD()
        logger.info("Silero VAD ONNX listo.")

        logger.info("Cargando VoiceClassifier (Resemblyzer + SVM)...")
        self.classifier = VoiceClassifier()
        logger.info("VoiceClassifier listo.")

    def make_vad_session(self) -> SileroVAD:
        """
        Crea una instancia VAD con estado propio para una sesion de llamada.
        Cada llamada necesita su propio estado (h, c) para no interferir.
        """
        vad = SileroVAD.__new__(SileroVAD)
        vad.session = self._vad_base.session  # Comparte el grafo ONNX (read-only)
        vad._reset_state()
        return vad

    def detect_beep(self, audio_bytes: bytes, sample_rate: int = AUDIO_INPUT_SAMPLE_RATE) -> dict:
        """
        Detecta pitido/tono tipico de buzon de voz usando FFT.
        Rapido: ~5ms. Se ejecuta antes que VAD y Resemblyzer.
        """
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

            if len(audio_np) < 1024:
                return {"detected": False, "reason": "Audio muy corto para detectar beep"}

            audio_np /= 32768.0

            fft_magnitude = np.abs(np.fft.rfft(audio_np))
            freqs = np.fft.rfftfreq(len(audio_np), 1.0 / sample_rate)

            beep_mask = (freqs >= BEEP_FREQ_MIN) & (freqs <= BEEP_FREQ_MAX)
            beep_energy = np.sum(fft_magnitude[beep_mask])
            total_energy = np.sum(fft_magnitude)

            if total_energy == 0:
                return {"detected": False, "reason": "Sin energia de audio"}

            beep_ratio = beep_energy / total_energy

            beep_mags = fft_magnitude.copy()
            beep_mags[~beep_mask] = 0
            dominant_idx = np.argmax(beep_mags)
            dominant_freq = freqs[dominant_idx] if beep_mags[dominant_idx] > 0 else 0

            if beep_ratio > BEEP_ENERGY_THRESHOLD and dominant_freq > 0:
                peak_energy = fft_magnitude[dominant_idx]
                purity = peak_energy / beep_energy if beep_energy > 0 else 0
                confidence = min(0.95, beep_ratio * purity * 2)

                if confidence >= BEEP_MIN_CONFIDENCE:
                    logger.info(
                        f"BEEP detectado: freq={dominant_freq:.0f}Hz, "
                        f"ratio={beep_ratio:.2f}, conf={confidence:.2f}"
                    )
                    return {
                        "detected": True,
                        "frequency": float(dominant_freq),
                        "confidence": float(confidence),
                        "reason": f"Tono detectado a {dominant_freq:.0f}Hz"
                    }

            return {
                "detected": False,
                "frequency": float(dominant_freq) if dominant_freq > 0 else 0,
                "energy_ratio": float(beep_ratio),
                "reason": "No se detecto tono de beep"
            }

        except Exception as e:
            logger.error(f"Error en deteccion de beep: {e}")
            return {"detected": False, "reason": f"Error: {str(e)}"}

    def classify_audio(self, audio_np: np.ndarray) -> dict:
        """Clasifica audio acumulado como HUMAN o MACHINE via Resemblyzer + SVM."""
        return self.classifier.predict(audio_np, sample_rate=VAD_SAMPLE_RATE)


class AMDSession:
    """
    Sesion AMD para una llamada individual.
    Cada sesion tiene su propio estado VAD para no interferir con otras llamadas.
    """

    def __init__(self, detector: AMDDetector, call_id: str, sample_rate: int = AUDIO_INPUT_SAMPLE_RATE):
        self.detector = detector
        self.call_id = call_id
        self.input_sample_rate = sample_rate

        # VAD con estado propio para esta sesion (no interfiere con otras llamadas)
        self._vad = detector.make_vad_session()

        # Buffer de audio acumulado a 16kHz (para Resemblyzer)
        self._audio_buffer_16k = np.array([], dtype=np.float32)

        # Segundos de voz activa acumulados
        self._voice_seconds = 0.0

        # Segundos de audio total procesados
        self._total_seconds = 0.0

        # Estado de decision
        self._decision_made = False
        self._final_result = None

        # Resample ratio entrada -> 16kHz
        self._needs_resample = (sample_rate != VAD_SAMPLE_RATE)
        self._resample_ratio = VAD_SAMPLE_RATE / sample_rate

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """Convierte PCM 16-bit bytes a float32 normalizado [-1, 1]"""
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _resample_to_16k(self, audio_np: np.ndarray) -> np.ndarray:
        """Resamplea audio a 16kHz si es necesario."""
        if not self._needs_resample:
            return audio_np
        target_len = int(len(audio_np) * self._resample_ratio)
        return scipy.signal.resample(audio_np, target_len).astype(np.float32)

    def process_audio(self, audio_bytes: bytes) -> dict | None:
        """
        Procesa un chunk de audio.

        Flujo:
        1. Detectar beep (FFT ~5ms) -> MACHINE inmediato si hay tono
        2. Convertir a float32 y resamplear a 16kHz
        3. Correr Silero VAD ONNX en chunks de 512 samples
        4. Acumular audio en buffer
        5. Si hay >= AMD_MIN_VOICE_SECONDS de voz -> clasificar con Resemblyzer+SVM
        6. Si total >= AMD_FALLBACK_SECONDS sin decision -> clasificar con lo que hay
        7. Si silencio total en fallback -> MACHINE directo
        """
        if self._decision_made:
            return self._final_result

        # PASO 1: Deteccion de beep
        beep = self.detector.detect_beep(audio_bytes, self.input_sample_rate)
        if beep.get("detected"):
            return self._make_decision(
                result="MACHINE",
                confidence=beep["confidence"],
                reason=f"Pitido de buzon detectado a {beep.get('frequency', 0):.0f}Hz",
                forced=False
            )

        # PASO 2: Convertir y resamplear
        audio_f32 = self._bytes_to_float32(audio_bytes)
        audio_16k = self._resample_to_16k(audio_f32)

        self._audio_buffer_16k = np.concatenate([self._audio_buffer_16k, audio_16k])
        self._total_seconds += len(audio_f32) / self.input_sample_rate

        # PASO 3: VAD en chunks de 512 samples
        voice_samples = 0
        offset = 0

        while offset + VAD_CHUNK_SAMPLES <= len(audio_16k):
            chunk = audio_16k[offset: offset + VAD_CHUNK_SAMPLES]
            prob = self._vad(chunk, VAD_SAMPLE_RATE)
            if prob >= VAD_THRESHOLD:
                voice_samples += VAD_CHUNK_SAMPLES
            offset += VAD_CHUNK_SAMPLES

        self._voice_seconds += voice_samples / VAD_SAMPLE_RATE

        logger.debug(
            f"[{self.call_id}] total={self._total_seconds:.2f}s "
            f"voz={self._voice_seconds:.2f}s"
        )

        # PASO 4: Clasificar si hay suficiente voz
        if self._voice_seconds >= AMD_MIN_VOICE_SECONDS:
            logger.info(
                f"[{self.call_id}] Voz suficiente ({self._voice_seconds:.2f}s) -> clasificando"
            )
            return self._classify_and_decide(forced=False)

        # PASO 5: Fallback por tiempo total
        if self._total_seconds >= AMD_FALLBACK_SECONDS:
            logger.info(
                f"[{self.call_id}] Fallback ({self._total_seconds:.2f}s) -> clasificando"
            )
            if self._voice_seconds < 0.3:
                return self._make_decision(
                    result="MACHINE",
                    confidence=0.75,
                    reason=f"Silencio prolongado ({self._total_seconds:.1f}s sin voz)",
                    forced=True
                )
            return self._classify_and_decide(forced=True)

        return None

    def force_decision(self) -> dict:
        """Fuerza decision final por timeout del servidor."""
        if self._decision_made:
            return self._final_result

        logger.info(
            f"[{self.call_id}] force_decision: "
            f"total={self._total_seconds:.2f}s, voz={self._voice_seconds:.2f}s"
        )

        if self._voice_seconds < 0.3 or len(self._audio_buffer_16k) < VAD_SAMPLE_RATE * 0.5:
            return self._make_decision(
                result="MACHINE",
                confidence=0.70,
                reason="Timeout: sin voz suficiente para clasificar",
                forced=True
            )

        return self._classify_and_decide(forced=True)

    def _classify_and_decide(self, forced: bool) -> dict | None:
        """Llama al clasificador Resemblyzer+SVM y decide."""
        classification = self.detector.classify_audio(self._audio_buffer_16k)

        result = classification.get("result", "UNKNOWN")
        confidence = classification.get("confidence", 0.0)
        reason = classification.get("reason", "")

        # Si confianza baja y no estamos forzados, pedir mas audio
        if not forced and confidence < AMD_CONFIDENCE_THRESHOLD and result != "UNKNOWN":
            logger.info(
                f"[{self.call_id}] Confianza baja ({confidence:.2f}), esperando mas audio"
            )
            return None

        # UNKNOWN = SVM no entrenado u error -> MACHINE por seguridad
        if result == "UNKNOWN":
            result = "MACHINE"
            confidence = 0.60
            reason = "No se pudo clasificar, asumiendo MACHINE por seguridad"

        return self._make_decision(result=result, confidence=confidence, reason=reason, forced=forced)

    def _make_decision(self, result: str, confidence: float, reason: str, forced: bool) -> dict:
        """Registra y retorna la decision final."""
        self._decision_made = True
        self._final_result = {
            "call_id": self.call_id,
            "result": result,
            "confidence": round(confidence, 4),
            "reason": reason,
            "forced": forced,
            "voice_seconds": round(self._voice_seconds, 2),
            "total_seconds": round(self._total_seconds, 2),
        }
        logger.info(f"[{self.call_id}] DECISION: {self._final_result}")
        return self._final_result