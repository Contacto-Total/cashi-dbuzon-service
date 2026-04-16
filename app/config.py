import os

# ============================================================
# MODELO SVM
# ============================================================
# Ruta donde se guarda/carga el modelo SVM entrenado
SVM_MODEL_PATH = os.getenv("SVM_MODEL_PATH", "models/svm_model.pkl")

# ============================================================
# CONFIGURACION AMD
# ============================================================

# Tiempo maximo total para tomar decision (segundos)
AMD_DECISION_TIMEOUT_SECONDS = float(os.getenv("AMD_TIMEOUT", "3.0"))

# Segundos de audio con voz que necesita Resemblyzer para clasificar con precision
AMD_MIN_VOICE_SECONDS = float(os.getenv("AMD_MIN_VOICE_SECONDS", "1.0"))

# Si VAD no detecta voz clara en este tiempo, acumula y manda todo a Resemblyzer
AMD_FALLBACK_SECONDS = float(os.getenv("AMD_FALLBACK_SECONDS", "3.0"))

# Umbral de confianza del SVM para aceptar una decision (0.0 - 1.0)
AMD_CONFIDENCE_THRESHOLD = float(os.getenv("AMD_CONFIDENCE_THRESHOLD", "0.70"))

# ============================================================
# SILERO VAD
# ============================================================

# Sample rate de trabajo interno (Silero VAD opera a 16000Hz)
VAD_SAMPLE_RATE = 16000

# Sample rate del audio entrante desde FreeSwitch (tipicamente 8000Hz)
AUDIO_INPUT_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "8000"))

# Umbral de probabilidad de voz para Silero VAD (0.0 - 1.0)
# Mas alto = mas estricto, menos falsos positivos
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

# Tamano del chunk de audio que procesa VAD (samples a 16kHz)
# 512 samples = 32ms a 16kHz — optimo para Silero
VAD_CHUNK_SAMPLES = 512

# ============================================================
# DETECCION DE BEEP / TONO
# ============================================================

# Rango de frecuencias tipicas de beeps de buzon de voz (Hz)
BEEP_FREQ_MIN = 800
BEEP_FREQ_MAX = 1800

# Proporcion minima de energia en rango de beep para considerar tono
BEEP_ENERGY_THRESHOLD = 0.35

# Confianza minima para aceptar deteccion de beep
BEEP_MIN_CONFIDENCE = 0.40

# ============================================================
# SERVIDOR
# ============================================================
SERVER_HOST = os.getenv("AMD_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("AMD_PORT", "8765"))