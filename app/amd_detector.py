"""
AMD Detector - Deteccion de Buzon de Voz
Usa Silero VAD + Resemblyzer + SVM (sin transcripcion)
"""
import logging
import numpy as np
import torch
import scipy.signal
from collections import deque

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


class AMDDetector:
    """
    Detector principal AMD.
    Singleton compartido entre todas las sesiones.
    Contiene: modelo VAD de Silero y clasificador Resemblyzer+SVM.
    """

    def __init__(self):
        logger.info("Cargando Silero VAD...")
        # Silero VAD se descarga automaticamente desde torch.hub la primera vez
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False
        )
        self.vad_model.eval()
        logger.info("Silero VAD cargado.")

        logger.info("Cargando VoiceClassifier (Resemblyzer + SVM)...")
        self.classifier = VoiceClassifier()
        logger.info("VoiceClassifier listo.")

    def detect_beep(self, audio_bytes: bytes, sample_rate: int = AUDIO_INPUT_SAMPLE_RATE) -> dict:
        """
        Detecta pitido/tono tipico de buzon de voz usando FFT.
        Rapido: ~5-10ms. Se ejecuta antes que VAD y Resemblyzer.

        Args:
            audio_bytes: Audio PCM 16-bit signed
            sample_rate: Sample rate del audio entrante

        Returns:
            dict con detected (bool), frequency, confidence
        """
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

            if len(audio_np) < 1024:
                return {"detected": False, "reason": "Audio muy corto para detectar beep"}

            # Normalizar
            audio_np /= 32768.0

            # FFT
            fft_magnitude = np.abs(np.fft.rfft(audio_np))
            freqs = np.fft.rfftfreq(len(audio_np), 1.0 / sample_rate)

            # Energia en rango de beep
            beep_mask = (freqs >= BEEP_FREQ_MIN) & (freqs <= BEEP_FREQ_MAX)
            beep_energy = np.sum(fft_magnitude[beep_mask])
            total_energy = np.sum(fft_magnitude)

            if total_energy == 0:
                return {"detected": False, "reason": "Sin energia de audio"}

            beep_ratio = beep_energy / total_energy

            # Frecuencia dominante en el rango de beep
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

    def run_vad(self, audio_np_16k: np.ndarray) -> float:
        """
        Ejecuta Silero VAD sobre un chunk de audio a 16kHz.

        Args:
            audio_np_16k: Audio float32 a 16kHz, shape (VAD_CHUNK_SAMPLES,)

        Returns:
            Probabilidad de voz (0.0 - 1.0)
        """
        with torch.no_grad():
            tensor = torch.FloatTensor(audio_np_16k).unsqueeze(0)
            prob = self.vad_model(tensor, VAD_SAMPLE_RATE).item()
        return prob

    def classify_audio(self, audio_np: np.ndarray) -> dict:
        """
        Clasifica audio acumulado como HUMAN o MACHINE.

        Args:
            audio_np: Audio float32 a VAD_SAMPLE_RATE (16kHz)

        Returns:
            dict con result, confidence, reason
        """
        return self.classifier.predict(audio_np, sample_rate=VAD_SAMPLE_RATE)


class AMDSession:
    """
    Sesion AMD para una llamada individual.
    Recibe chunks de audio, corre VAD en tiempo real,
    y cuando hay suficiente voz llama al clasificador.
    """

    def __init__(self, detector: AMDDetector, call_id: str, sample_rate: int = AUDIO_INPUT_SAMPLE_RATE):
        self.detector = detector
        self.call_id = call_id
        self.input_sample_rate = sample_rate

        # Buffer de audio acumulado a 16kHz (para VAD y Resemblyzer)
        self._audio_buffer_16k = np.array([], dtype=np.float32)

        # Buffer de chunks crudos pendientes de convertir a 16kHz (en bytes de entrada)
        self._raw_buffer = bytearray()

        # Cuantos segundos de voz activa llevamos acumulados
        self._voice_seconds = 0.0

        # Cuantos segundos de audio total hemos procesado
        self._total_seconds = 0.0

        # Estado: si ya tomamos decision
        self._decision_made = False
        self._final_result = None

        # Resample ratio: entrada / salida
        # Silero VAD necesita 16kHz internamente
        self._needs_resample = (sample_rate != VAD_SAMPLE_RATE)
        self._resample_ratio = VAD_SAMPLE_RATE / sample_rate

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """Convierte PCM 16-bit bytes a float32 normalizado [-1, 1]"""
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _resample_to_16k(self, audio_np: np.ndarray) -> np.ndarray:
        """Resamplea audio al sample rate de VAD (16kHz) si es necesario"""
        if not self._needs_resample:
            return audio_np
        target_len = int(len(audio_np) * self._resample_ratio)
        return scipy.signal.resample(audio_np, target_len).astype(np.float32)

    def process_audio(self, audio_bytes: bytes) -> dict | None:
        """
        Procesa un chunk de audio.

        Flujo:
        1. Detectar beep → si hay beep = MACHINE inmediato
        2. Convertir a float32 y resamplear a 16kHz
        3. Correr Silero VAD en chunks de 512 samples
        4. Acumular audio con voz
        5. Si hay >= AMD_MIN_VOICE_SECONDS de voz → clasificar
        6. Si total >= AMD_FALLBACK_SECONDS sin decision → clasificar con lo que hay

        Args:
            audio_bytes: Chunk de audio PCM 16-bit

        Returns:
            dict con resultado si se tomo decision, None si necesita mas audio
        """
        if self._decision_made:
            return self._final_result

        # ── PASO 1: Deteccion de beep (rapidisima) ──────────────────────────
        beep = self.detector.detect_beep(audio_bytes, self.input_sample_rate)
        if beep.get("detected"):
            return self._make_decision(
                result="MACHINE",
                confidence=beep["confidence"],
                reason=f"Pitido de buzon detectado a {beep.get('frequency', 0):.0f}Hz",
                forced=False
            )

        # ── PASO 2: Convertir y resamplear ──────────────────────────────────
        audio_f32 = self._bytes_to_float32(audio_bytes)
        audio_16k = self._resample_to_16k(audio_f32)

        # Acumular en buffer global
        self._audio_buffer_16k = np.concatenate([self._audio_buffer_16k, audio_16k])
        chunk_duration = len(audio_f32) / self.input_sample_rate
        self._total_seconds += chunk_duration

        # ── PASO 3: Correr VAD en chunks de 512 samples ─────────────────────
        buffer_offset = 0
        voice_samples_this_chunk = 0

        while buffer_offset + VAD_CHUNK_SAMPLES <= len(audio_16k):
            vad_chunk = audio_16k[buffer_offset: buffer_offset + VAD_CHUNK_SAMPLES]
            voice_prob = self.detector.run_vad(vad_chunk)

            if voice_prob >= VAD_THRESHOLD:
                voice_samples_this_chunk += VAD_CHUNK_SAMPLES

            buffer_offset += VAD_CHUNK_SAMPLES

        # Calcular segundos de voz en este chunk
        voice_seconds_this_chunk = voice_samples_this_chunk / VAD_SAMPLE_RATE
        self._voice_seconds += voice_seconds_this_chunk

        logger.debug(
            f"[{self.call_id}] total={self._total_seconds:.2f}s "
            f"voz={self._voice_seconds:.2f}s "
            f"buffer_16k={len(self._audio_buffer_16k)} samples"
        )

        # ── PASO 4: Clasificar si hay suficiente voz ────────────────────────
        if self._voice_seconds >= AMD_MIN_VOICE_SECONDS:
            logger.info(
                f"[{self.call_id}] Voz suficiente ({self._voice_seconds:.2f}s) → clasificando"
            )
            return self._classify_and_decide(forced=False)

        # ── PASO 5: Fallback por tiempo total sin decision ───────────────────
        if self._total_seconds >= AMD_FALLBACK_SECONDS:
            logger.info(
                f"[{self.call_id}] Fallback por tiempo ({self._total_seconds:.2f}s) → clasificando"
            )
            # Si hay muy poca o nula voz en 3 segundos = silencio = MACHINE
            if self._voice_seconds < 0.3:
                return self._make_decision(
                    result="MACHINE",
                    confidence=0.75,
                    reason=f"Silencio prolongado ({self._total_seconds:.1f}s sin voz detectada)",
                    forced=True
                )
            return self._classify_and_decide(forced=True)

        return None  # Necesita mas audio

    def force_decision(self) -> dict:
        """
        Fuerza decision final (llamado por timeout del servidor).
        Clasifica con el audio acumulado o retorna MACHINE si hay silencio.
        """
        if self._decision_made:
            return self._final_result

        logger.info(
            f"[{self.call_id}] force_decision: "
            f"total={self._total_seconds:.2f}s, voz={self._voice_seconds:.2f}s"
        )

        # Sin voz significativa = MACHINE (buzon que no habló o silencio)
        if self._voice_seconds < 0.3 or len(self._audio_buffer_16k) < VAD_SAMPLE_RATE * 0.5:
            return self._make_decision(
                result="MACHINE",
                confidence=0.70,
                reason="Timeout: sin voz suficiente para clasificar",
                forced=True
            )

        return self._classify_and_decide(forced=True)

    def _classify_and_decide(self, forced: bool) -> dict:
        """Llama al clasificador y registra la decision."""
        classification = self.detector.classify_audio(self._audio_buffer_16k)

        result = classification.get("result", "UNKNOWN")
        confidence = classification.get("confidence", 0.0)
        reason = classification.get("reason", "")

        # Si la confianza es baja y no estamos forzados, pedir mas audio
        if not forced and confidence < AMD_CONFIDENCE_THRESHOLD and result != "UNKNOWN":
            logger.info(
                f"[{self.call_id}] Confianza baja ({confidence:.2f}), esperando mas audio"
            )
            return None

        # Si resultado es UNKNOWN (SVM no entrenado u otro error) = MACHINE por seguridad
        if result == "UNKNOWN":
            result = "MACHINE"
            confidence = 0.60
            reason = "No se pudo clasificar con certeza, asumiendo MACHINE por seguridad"

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