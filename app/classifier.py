"""
Classifier - SVM sobre embeddings de Resemblyzer
Carga el modelo entrenado y expone predict()
"""
import logging
import numpy as np
import joblib
import os
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

from app.config import SVM_MODEL_PATH, VAD_SAMPLE_RATE

logger = logging.getLogger(__name__)


class VoiceClassifier:
    """
    Clasificador de voz: HUMAN vs MACHINE
    Usa Resemblyzer para generar embeddings de 256 dimensiones
    y un SVM entrenado para clasificar.
    """

    def __init__(self):
        logger.info("Cargando VoiceEncoder (Resemblyzer)...")
        self.encoder = VoiceEncoder(device="cpu")
        logger.info("VoiceEncoder cargado.")

        self.svm = None
        self._load_svm()

    def _load_svm(self):
        """Carga el modelo SVM desde disco si existe"""
        if os.path.exists(SVM_MODEL_PATH):
            logger.info(f"Cargando modelo SVM desde: {SVM_MODEL_PATH}")
            self.svm = joblib.load(SVM_MODEL_PATH)
            logger.info("Modelo SVM cargado exitosamente.")
        else:
            logger.warning(
                f"Modelo SVM no encontrado en: {SVM_MODEL_PATH}. "
                "Ejecuta training/train.py para entrenar el modelo."
            )

    def is_ready(self) -> bool:
        """True si el SVM esta cargado y listo para clasificar"""
        return self.svm is not None

    def get_embedding(self, audio_np: np.ndarray, sample_rate: int = VAD_SAMPLE_RATE) -> np.ndarray:
        """
        Genera embedding de 256 dimensiones a partir de audio numpy.

        Args:
            audio_np: Audio como array float32 normalizado [-1, 1]
            sample_rate: Sample rate del audio (Resemblyzer espera 16000Hz)

        Returns:
            np.ndarray de shape (256,)
        """
        try:
            # Resemblyzer espera float32 a 16kHz
            wav = preprocess_wav(audio_np, source_sr=sample_rate)
            embedding = self.encoder.embed_utterance(wav)
            return embedding
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None

    def predict(self, audio_np: np.ndarray, sample_rate: int = VAD_SAMPLE_RATE) -> dict:
        """
        Clasifica audio como HUMAN o MACHINE.

        Args:
            audio_np: Audio como array float32 normalizado [-1, 1]
            sample_rate: Sample rate del audio

        Returns:
            dict con result, confidence, reason
        """
        if not self.is_ready():
            logger.error("SVM no cargado, no se puede clasificar.")
            return {
                "result": "UNKNOWN",
                "confidence": 0.0,
                "reason": "Modelo SVM no entrenado"
            }

        embedding = self.get_embedding(audio_np, sample_rate)

        if embedding is None:
            return {
                "result": "UNKNOWN",
                "confidence": 0.0,
                "reason": "Error generando embedding de voz"
            }

        try:
            # Reshape para sklearn (1 sample, 256 features)
            X = embedding.reshape(1, -1)

            # Prediccion con probabilidades
            proba = self.svm.predict_proba(X)[0]
            classes = self.svm.classes_  # ["HUMAN", "MACHINE"] o ["MACHINE", "HUMAN"]

            # Encontrar la clase con mayor probabilidad
            best_idx = int(np.argmax(proba))
            result = classes[best_idx]
            confidence = float(proba[best_idx])

            # Probabilidad de cada clase para logging
            proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
            logger.info(f"Clasificacion SVM: {result} (conf={confidence:.2f}) | probs={proba_dict}")

            return {
                "result": result,
                "confidence": confidence,
                "reason": f"Resemblyzer+SVM: {result} con {confidence:.0%} de confianza",
                "probabilities": proba_dict
            }

        except Exception as e:
            logger.error(f"Error en prediccion SVM: {e}")
            return {
                "result": "UNKNOWN",
                "confidence": 0.0,
                "reason": f"Error en clasificacion: {str(e)}"
            }