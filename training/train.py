"""
Script de entrenamiento del clasificador SVM
Uso:
    python training/train.py

Estructura de carpetas requerida:
    training/samples/human/    → audios .wav de personas reales
    training/samples/machine/  → audios .wav de buzones de voz

El script:
1. Carga todos los audios de cada carpeta
2. Genera embeddings con Resemblyzer
3. Entrena un SVM con los embeddings
4. Guarda el modelo en models/svm_model.pkl
5. Muestra metricas de precision en consola
"""
import os
import sys
import logging
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import librosa

# ── Configuracion ─────────────────────────────────────────────────────────────
SAMPLES_DIR = Path("training/samples")
HUMAN_DIR = SAMPLES_DIR / "human"
MACHINE_DIR = SAMPLES_DIR / "machine"
MODEL_OUTPUT_PATH = Path("models/svm_model.pkl")
TARGET_SR = 16000  # Resemblyzer opera a 16kHz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray | None:
    """
    Carga un archivo de audio y lo convierte a float32 mono a target_sr.
    Soporta: WAV, MP3, OGG, FLAC, etc.
    """
    try:
        # librosa carga y resamplea automaticamente
        audio, sr = librosa.load(
                str(path),
                sr=target_sr,
                mono=True,
                res_type="kaiser_fast"
            )
        if len(audio) < target_sr * 0.5:  # Minimo 0.5s de audio
            logger.warning(f"Audio muy corto (<0.5s): {path.name}")
            return None
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Error cargando {path.name}: {e}")
        return None


def get_embedding(encoder: VoiceEncoder, audio: np.ndarray) -> np.ndarray | None:
    """Genera embedding de 256 dimensiones con Resemblyzer."""
    try:
        wav = preprocess_wav(audio, source_sr=TARGET_SR)
        return encoder.embed_utterance(wav)
    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        return None


def load_samples(directory: Path, label: str, encoder: VoiceEncoder) -> tuple[list, list]:
    """
    Carga todos los audios de un directorio y genera sus embeddings.
    Retorna (embeddings, labels)
    """
    embeddings = []
    labels = []

    audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"}
    files = [f for f in directory.iterdir() if f.suffix.lower() in audio_extensions]

    if not files:
        logger.warning(f"No se encontraron audios en: {directory}")
        return embeddings, labels

    logger.info(f"Procesando {len(files)} archivos de '{label}'...")

    for i, audio_path in enumerate(files, 1):
        audio = load_audio(audio_path)
        if audio is None:
            continue

        embedding = get_embedding(encoder, audio)
        if embedding is None:
            continue

        embeddings.append(embedding)
        labels.append(label)

        if i % 10 == 0:
            logger.info(f"  {i}/{len(files)} procesados...")

    logger.info(f"  ✓ {len(embeddings)}/{len(files)} embeddings generados para '{label}'")
    return embeddings, labels


def train():
    """Entrena el clasificador SVM y lo guarda en disco."""

    # ── Verificar carpetas ────────────────────────────────────────────────────
    for directory in [HUMAN_DIR, MACHINE_DIR]:
        if not directory.exists():
            logger.error(f"Carpeta no encontrada: {directory}")
            logger.error("Crea las carpetas y agrega audios .wav:")
            logger.error("  training/samples/human/   → audios de personas reales")
            logger.error("  training/samples/machine/ → audios de buzones de voz")
            sys.exit(1)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Cargar VoiceEncoder ───────────────────────────────────────────────────
    logger.info("Cargando VoiceEncoder (Resemblyzer)...")
    encoder = VoiceEncoder(device="cpu")
    logger.info("VoiceEncoder listo.")

    # ── Cargar muestras ───────────────────────────────────────────────────────
    human_emb, human_lbl = load_samples(HUMAN_DIR, "HUMAN", encoder)
    machine_emb, machine_lbl = load_samples(MACHINE_DIR, "MACHINE", encoder)

    all_embeddings = human_emb + machine_emb
    all_labels = human_lbl + machine_lbl

    if len(all_embeddings) < 10:
        logger.error(f"Muy pocas muestras ({len(all_embeddings)}). Necesitas al menos 10 en total.")
        sys.exit(1)

    logger.info(f"\nTotal muestras: {len(all_embeddings)}")
    logger.info(f"  HUMAN:   {len(human_emb)}")
    logger.info(f"  MACHINE: {len(machine_emb)}")

    X = np.array(all_embeddings)
    y = np.array(all_labels)

    # ── Entrenar SVM ──────────────────────────────────────────────────────────
    logger.info("\nEntrenando SVM...")

    # Pipeline: normalizar embeddings + SVM con kernel RBF
    # probability=True es necesario para predict_proba()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",  # Maneja desbalance de clases
            random_state=42,
        ))
    ])

    # ── Cross-validation (5-fold) ─────────────────────────────────────────────
    logger.info("Evaluando con cross-validation 5-fold...")
    n_splits = min(5, len(human_emb), len(machine_emb))

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        logger.info(f"Accuracy CV: {scores.mean():.2%} ± {scores.std():.2%}")
        logger.info(f"Scores por fold: {[f'{s:.2%}' for s in scores]}")
    else:
        logger.warning("Muy pocas muestras para cross-validation, entrenando directamente.")

    # ── Entrenar modelo final con todos los datos ─────────────────────────────
    pipeline.fit(X, y)

    # ── Metricas en training set ──────────────────────────────────────────────
    y_pred = pipeline.predict(X)
    logger.info("\nReporte de clasificacion (training set):")
    logger.info("\n" + classification_report(y, y_pred))
    logger.info("Matriz de confusion:")
    logger.info(f"\n{confusion_matrix(y, y_pred)}")

    # ── Guardar modelo ────────────────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    logger.info(f"\n✓ Modelo guardado en: {MODEL_OUTPUT_PATH}")
    logger.info("Listo. Reinicia el servicio para cargar el nuevo modelo.")


if __name__ == "__main__":
    train()