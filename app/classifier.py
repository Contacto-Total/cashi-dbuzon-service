"""
Classifier - Whisper + keyword matching
Transcribe el audio acumulado y decide HUMAN/MACHINE por contenido.
Reemplaza al SVM Resemblyzer que no generalizaba con poca data.
"""
import logging
import numpy as np
import scipy.signal

from app.config import VAD_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Palabras clave que indican un saludo de buzon de voz en espanol (LATAM / Peru).
# Se busca cualquier substring dentro del texto transcrito (case-insensitive).
#
# Operadores peruanos (Movistar, Claro, Entel, Bitel) usan frases tipicas:
# - "tu llamada sera transferida a una casilla de voz"
# - "deja tu mensaje en la casilla de voz"
# - "despues de la señal / presione cualquier tecla"
# Whisper-tiny a veces confunde "casilla" -> "cacilla" / "cassiye" / "cacilla":
# agregamos variantes para absorber esos errores de ASR.
MACHINE_KEYWORDS = [
    # Frases de transferencia a buzon
    "casilla", "cacilla", "cassiye", "casiya", "cacilya",
    "transferida", "transferido",
    "llamada sera", "llamada será",
    # Frases de deje mensaje
    "mensaje",
    "buzon", "buzón",
    "deja tu", "déjate", "dejate",
    "deje",
    "grabar", "grabe",
    # Frases de tono/señal
    "tono",
    "señal", "senal",
    "tecla", "decla", "decala", "decada", "década",
    "presione", "precione", "preció", "precio",
    "cualquier",  # "presione cualquier tecla" -> muy especifico de buzones
    "terminar",
    "despues del", "después del",
    "despues de", "después de",
    # Frases de indisponibilidad
    "no se encuentra",
    "no esta disponible", "no está disponible",
    "no disponible",
    "numero marcado", "número marcado",
    "desvio", "desvío",
    "contestador",
    "apagado",
    "fuera de",
    "cobertura",
    "no atiende",
    "ahora no puede",
]

# Palabras tipicas de humano contestando el telefono.
HUMAN_KEYWORDS = [
    "alo", "aló",
    "hola",
    "diga",
    "bueno",
    "quien habla", "quién habla",
    "con quien", "con quién",
    "si diga", "sí diga",
]


class VoiceClassifier:
    """
    Clasificador HUMAN vs MACHINE basado en transcripcion con Whisper
    y matching de keywords tipicos de buzones y humanos.
    """

    def __init__(self):
        logger.info("Cargando Whisper-tiny (faster-whisper)...")
        try:
            from faster_whisper import WhisperModel
            # int8 para correr rapido en CPU, ~40MB modelo + ~100MB runtime
            self.model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
            )
            logger.info("Whisper-tiny cargado.")
        except Exception as e:
            logger.error(f"Error cargando Whisper: {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def _resample_to_16k(self, audio_np: np.ndarray, source_sr: int) -> np.ndarray:
        """Whisper espera float32 a 16kHz mono."""
        if source_sr == 16000:
            return audio_np.astype(np.float32)
        gcd = np.gcd(source_sr, 16000)
        up = 16000 // gcd
        down = source_sr // gcd
        return scipy.signal.resample_poly(audio_np, up, down).astype(np.float32)

    def predict(self, audio_np: np.ndarray, sample_rate: int = VAD_SAMPLE_RATE) -> dict:
        """
        Transcribe el audio y decide HUMAN/MACHINE por keywords.

        Returns:
            dict con: result, confidence, reason, transcription
        """
        if not self.is_ready():
            return {
                "result": "UNKNOWN",
                "confidence": 0.0,
                "reason": "Whisper no cargado",
            }

        # Minimo 0.3s de audio para que Whisper tenga contexto util
        if len(audio_np) < sample_rate * 0.3:
            return {
                "result": "UNKNOWN",
                "confidence": 0.0,
                "reason": "Audio muy corto para transcribir",
            }

        try:
            audio_16k = self._resample_to_16k(audio_np, sample_rate)

            segments, _info = self.model.transcribe(
                audio_16k,
                language="es",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text for s in segments).lower().strip()

            logger.info(f"Whisper transcripcion: '{text[:150]}'")

            # Truncar texto al log para evitar reasons gigantes
            text_preview = text[:120].replace('"', "'")

            if not text:
                # Sin transcripcion: Whisper no oyo nada claro.
                # Default MACHINE por seguridad (evita pasar buzones silenciosos como humanos).
                return {
                    "result": "MACHINE",
                    "confidence": 0.65,
                    "reason": "Sin transcripcion (audio no claro)",
                    "transcription": "",
                }

            machine_hits = [kw for kw in MACHINE_KEYWORDS if kw in text]
            human_hits = [kw for kw in HUMAN_KEYWORDS if kw in text]

            if machine_hits:
                # Mas hits -> mas confianza, capped a 0.95
                confidence = min(0.95, 0.80 + 0.03 * len(machine_hits))
                return {
                    "result": "MACHINE",
                    "confidence": confidence,
                    "reason": f"Buzon kw={machine_hits} | texto=\"{text_preview}\"",
                    "transcription": text,
                }

            if human_hits:
                return {
                    "result": "HUMAN",
                    "confidence": 0.85,
                    "reason": f"Humano kw={human_hits} | texto=\"{text_preview}\"",
                    "transcription": text,
                }

            # Voz transcrita pero sin keywords conocidas. Un humano puede decir
            # algo fuera del diccionario ("ya te llamo", "espera", etc), en
            # cambio un buzon casi siempre incluye alguna frase estandar.
            return {
                "result": "HUMAN",
                "confidence": 0.60,
                "reason": f"Sin keywords | texto=\"{text_preview}\"",
                "transcription": text,
            }

        except Exception as e:
            logger.error(f"Error transcribiendo con Whisper: {e}")
            return {
                "result": "UNKNOWN",
                "confidence": 0.0,
                "reason": f"Error Whisper: {e}",
            }
