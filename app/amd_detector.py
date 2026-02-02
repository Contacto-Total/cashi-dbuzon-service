"""
AMD Detector - Deteccion de Buzon de Voz usando Vosk
"""
import json
import logging
import numpy as np
from vosk import Model, KaldiRecognizer
from app.config import (
    VOSK_MODEL_PATH,
    VOICEMAIL_KEYWORDS,
    AMD_MIN_SPEECH_FOR_MACHINE
)

logger = logging.getLogger(__name__)

class AMDDetector:
    """
    Detector de Maquina Contestadora (AMD)
    Usa Vosk para transcribir audio y detectar si es humano o maquina
    Incluye deteccion de pitidos/tonos para buzones de voz
    """

    # Rango de frecuencias tipicas de beeps de buzon (Hz)
    BEEP_FREQ_MIN = 800
    BEEP_FREQ_MAX = 1800
    # Umbral de energia para considerar que hay un tono
    BEEP_ENERGY_THRESHOLD = 0.3

    def __init__(self):
        logger.info(f"Cargando modelo Vosk desde: {VOSK_MODEL_PATH}")
        self.model = Model(VOSK_MODEL_PATH)
        logger.info("Modelo Vosk cargado exitosamente")

    def create_recognizer(self, sample_rate: int = 8000) -> KaldiRecognizer:
        """Crea un nuevo recognizer para una llamada"""
        return KaldiRecognizer(self.model, sample_rate)

    def detect_beep(self, audio_data: bytes, sample_rate: int = 8000) -> dict:
        """
        Detecta si hay un pitido/tono tipico de buzon de voz en el audio.
        Usa FFT para analizar las frecuencias dominantes.

        Args:
            audio_data: Bytes de audio (16-bit PCM)
            sample_rate: Frecuencia de muestreo (default 8000Hz)

        Returns:
            dict con detected (bool), frequency (Hz), confidence (0-1)
        """
        try:
            # Convertir bytes a array de numpy (16-bit signed)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            if len(audio_array) < 1024:
                return {"detected": False, "reason": "Audio muy corto"}

            # Normalizar
            audio_array = audio_array / 32768.0

            # Aplicar FFT
            fft_result = np.fft.rfft(audio_array)
            fft_magnitude = np.abs(fft_result)

            # Calcular frecuencias correspondientes
            freqs = np.fft.rfftfreq(len(audio_array), 1.0 / sample_rate)

            # Buscar energia en el rango de frecuencias de beep
            beep_mask = (freqs >= self.BEEP_FREQ_MIN) & (freqs <= self.BEEP_FREQ_MAX)
            beep_energy = np.sum(fft_magnitude[beep_mask])
            total_energy = np.sum(fft_magnitude)

            if total_energy == 0:
                return {"detected": False, "reason": "Sin energia de audio"}

            # Proporcion de energia en el rango de beep
            beep_ratio = beep_energy / total_energy

            # Encontrar la frecuencia dominante en el rango de beep
            beep_magnitudes = fft_magnitude.copy()
            beep_magnitudes[~beep_mask] = 0
            dominant_idx = np.argmax(beep_magnitudes)
            dominant_freq = freqs[dominant_idx] if beep_magnitudes[dominant_idx] > 0 else 0

            # Verificar si es un tono puro (pico bien definido)
            if beep_ratio > self.BEEP_ENERGY_THRESHOLD and dominant_freq > 0:
                # Calcular que tan "puro" es el tono (concentracion de energia)
                peak_energy = fft_magnitude[dominant_idx]
                purity = peak_energy / beep_energy if beep_energy > 0 else 0

                confidence = min(0.95, beep_ratio * purity * 2)

                logger.info(f"BEEP detectado: freq={dominant_freq:.0f}Hz, ratio={beep_ratio:.2f}, purity={purity:.2f}, conf={confidence:.2f}")

                return {
                    "detected": True,
                    "frequency": float(dominant_freq),
                    "confidence": float(confidence),
                    "energy_ratio": float(beep_ratio),
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

    def analyze_transcription(self, text: str, speech_duration: float = 0) -> dict:
        """
        Analiza el texto transcrito para determinar si es humano o maquina

        Args:
            text: Texto transcrito del audio
            speech_duration: Duracion del habla en segundos

        Returns:
            dict con resultado: HUMAN, MACHINE, o UNKNOWN
        """
        text_lower = text.lower().strip()

        logger.info(f"Analizando: '{text_lower}' (duracion habla: {speech_duration:.2f}s)")

        # Si no hay texto = silencio
        # Silencio puede ser: cliente callado O buzón que aún no habla
        # Mejor esperar más audio antes de decidir
        if not text_lower:
            return {
                "result": "UNKNOWN",
                "confidence": 0.40,
                "reason": "Silencio - esperando más audio para determinar",
                "transcription": ""
            }

        # Contar palabras clave de buzon
        keywords_found = []
        for keyword in VOICEMAIL_KEYWORDS:
            if keyword in text_lower:
                keywords_found.append(keyword)

        # REGLA 1: Si encontro palabras clave de buzon = MAQUINA
        if len(keywords_found) >= 1:
            confidence = min(0.95, 0.7 + (len(keywords_found) * 0.1))
            return {
                "result": "MACHINE",
                "confidence": confidence,
                "reason": f"Palabras clave detectadas: {keywords_found}",
                "transcription": text_lower,
                "keywords": keywords_found
            }

        # REGLA 2: Si habla mucho tiempo continuo sin pausas = MAQUINA
        if speech_duration > AMD_MIN_SPEECH_FOR_MACHINE:
            # Contar palabras - un buzon tipicamente tiene muchas palabras
            word_count = len(text_lower.split())
            if word_count > 8:  # Mas de 8 palabras en los primeros segundos
                return {
                    "result": "MACHINE",
                    "confidence": 0.75,
                    "reason": f"Habla continua larga ({speech_duration:.1f}s, {word_count} palabras)",
                    "transcription": text_lower
                }

        # REGLA 3: Respuesta corta tipica de humano ("alo", "hola", "si", "digame")
        # Solo si es un saludo CONOCIDO y es corto
        human_greetings = ["alo", "aló", "hola", "si", "sí", "diga", "digame", "dígame",
                          "bueno", "quien", "quién", "mande"]

        words = text_lower.split()
        if len(words) <= 3:
            for greeting in human_greetings:
                if greeting in text_lower:
                    return {
                        "result": "HUMAN",
                        "confidence": 0.85,
                        "reason": f"Saludo humano detectado: '{text_lower}'",
                        "transcription": text_lower
                    }

        # REGLA 4: Palabras sueltas sin contexto = probablemente fragmento mal transcrito
        # Palabras como "archivo", "solo", "quieren", "no" son frecuentemente
        # fragmentos mal transcritos de buzones de voz, NO respuestas humanas reales.
        # Un humano real dice "aló", "hola", "sí", etc. - no palabras random.
        suspicious_single_words = [
            "archivo", "solo", "quieren", "león", "holgado", "luego",
            "lo", "el", "la", "los", "las", "un", "una", "de", "que",
            "le", "se", "no", "es", "en", "por", "con", "para"
        ]

        if len(words) == 1:
            single_word = words[0]
            # Si es una palabra suelta sospechosa = necesitamos más audio
            if single_word in suspicious_single_words or single_word not in human_greetings:
                return {
                    "result": "UNKNOWN",
                    "confidence": 0.40,
                    "reason": f"Palabra suelta sin contexto: '{text_lower}' - esperando mas audio",
                    "transcription": text_lower
                }

        # REGLA 5: Si es corto (2-4 palabras) y NO es saludo conocido = UNKNOWN
        # Dar más tiempo para escuchar el contexto completo
        if len(words) <= 4 and len(keywords_found) == 0:
            # Verificar si contiene al menos un saludo humano
            has_greeting = any(g in text_lower for g in human_greetings)
            if has_greeting:
                return {
                    "result": "HUMAN",
                    "confidence": 0.70,
                    "reason": f"Respuesta corta con saludo: '{text_lower}'",
                    "transcription": text_lower
                }
            else:
                # No tiene saludo conocido - probablemente fragmento de buzón
                return {
                    "result": "UNKNOWN",
                    "confidence": 0.50,
                    "reason": f"Respuesta corta sin saludo conocido: '{text_lower}' - esperando mas audio",
                    "transcription": text_lower
                }

        # Default: Si no podemos determinar con confianza
        return {
            "result": "UNKNOWN",
            "confidence": 0.5,
            "reason": "No se pudo determinar con certeza",
            "transcription": text_lower
        }


class AMDSession:
    """
    Sesion AMD para una llamada individual
    Acumula audio y toma decision
    """

    def __init__(self, detector: AMDDetector, call_id: str, sample_rate: int = 8000):
        self.detector = detector
        self.call_id = call_id
        self.sample_rate = sample_rate
        self.recognizer = detector.create_recognizer(sample_rate)
        self.accumulated_text = ""
        self.speech_start_time = None
        self.total_speech_duration = 0.0
        self.decision_made = False
        self.final_result = None

    def process_audio(self, audio_data: bytes) -> dict | None:
        """
        Procesa un chunk de audio

        Returns:
            dict con resultado si se tomo una decision, None si necesita mas audio
        """
        if self.decision_made:
            return self.final_result

        # Alimentar audio al recognizer
        if self.recognizer.AcceptWaveform(audio_data):
            # Resultado parcial completo
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")

            if text:
                self.accumulated_text += " " + text
                self.total_speech_duration += len(audio_data) / (self.sample_rate * 2)  # 16-bit = 2 bytes

                # Analizar lo que tenemos hasta ahora
                analysis = self.detector.analyze_transcription(
                    self.accumulated_text.strip(),
                    self.total_speech_duration
                )

                # Solo tomar decision si:
                # 1. Es MACHINE o HUMAN (no UNKNOWN)
                # 2. Tenemos confianza >= 70%
                if analysis["result"] != "UNKNOWN" and analysis["confidence"] >= 0.70:
                    self.decision_made = True
                    self.final_result = {
                        "call_id": self.call_id,
                        **analysis
                    }
                    logger.info(f"[{self.call_id}] Decision: {self.final_result}")
                    return self.final_result
        else:
            # Resultado parcial (en progreso)
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get("partial", "")

            # Analisis rapido del parcial para detectar buzones obvios
            if partial_text:
                quick_analysis = self.detector.analyze_transcription(partial_text, 0)
                if quick_analysis["result"] == "MACHINE" and quick_analysis["confidence"] >= 0.85:
                    self.decision_made = True
                    self.final_result = {
                        "call_id": self.call_id,
                        **quick_analysis,
                        "partial": True
                    }
                    logger.info(f"[{self.call_id}] Decision rapida: {self.final_result}")
                    return self.final_result

        return None

    def force_decision(self) -> dict:
        """
        Fuerza una decision con lo que se tiene (timeout)
        Si el resultado es UNKNOWN, opta por MACHINE (mas seguro no conectar buzon)
        """
        if self.decision_made:
            return self.final_result

        # Obtener texto final
        final = json.loads(self.recognizer.FinalResult())
        final_text = final.get("text", "")

        if final_text:
            self.accumulated_text += " " + final_text

        # Analizar todo lo acumulado
        analysis = self.detector.analyze_transcription(
            self.accumulated_text.strip(),
            self.total_speech_duration
        )

        # Si despues del timeout el resultado es UNKNOWN, optar por MACHINE
        # Es más seguro NO conectar un posible buzón a un agente
        # Un humano real generalmente dice algo reconocible (aló, hola, etc.)
        if analysis["result"] == "UNKNOWN":
            accumulated = self.accumulated_text.strip()
            # Si hay algo de texto pero no es saludo conocido = probable buzón
            # Si es silencio total = puede ser buzón que no grabó bien
            analysis = {
                "result": "MACHINE",
                "confidence": 0.60,
                "reason": f"Timeout sin saludo humano reconocido - probable buzón o sistema (texto: '{accumulated}')",
                "transcription": accumulated
            }

        self.decision_made = True
        self.final_result = {
            "call_id": self.call_id,
            **analysis,
            "forced": True
        }

        logger.info(f"[{self.call_id}] Decision forzada: {self.final_result}")
        return self.final_result
