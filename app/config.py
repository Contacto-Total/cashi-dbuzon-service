import os

# Ruta al modelo Vosk
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-es-0.42")

# Configuracion AMD
AMD_DECISION_TIMEOUT_SECONDS = 3.5  # Tiempo maximo para tomar decision
AMD_MIN_SPEECH_FOR_MACHINE = 2.5    # Si habla mas de 2.5s continuo = maquina

# Palabras clave que indican buzon de voz o sistema automatizado
VOICEMAIL_KEYWORDS = [
    # Espanol - Buzón de voz tradicional
    "mensaje", "buzón", "buzon", "tono", "ocupado", "disponible",
    "después del", "despues del", "deje su", "deja tu", "no se encuentra",
    "fuera de servicio", "no está disponible", "no esta disponible",
    "vuelva a llamar", "intentelo más tarde", "intentelo mas tarde",
    "número que usted marcó", "numero que usted marco",
    "en este momento", "por favor", "gracias por llamar",
    "horario de atención", "horario de atencion",
    "marque la extensión", "marque la extension",
    "bienvenido", "ha comunicado con", "ha llamado a",
    # Buzón lleno / capacidad máxima
    "capacidad máxima", "capacidad maxima", "alcanzó la capacidad",
    "alcanzo la capacidad", "buzón lleno", "buzon lleno",
    "no puede recibir", "más mensajes", "mas mensajes",
    "alcanzó su", "alcanzo su", "está lleno", "esta lleno",
    # Buzón de voz personalizado
    "dejas tu nombre", "deja tu nombre", "tu nombre y",
    "motivo de tu llamada", "motivo de la llamada",
    "te llamaremos", "te devolvemos", "devolvere la llamada",
    "no puedo atender", "no puedo contestar", "estoy ocupado",
    "deja tu mensaje", "dejar un mensaje",
    # Operadora / Sistema telefónico
    "el número marcado", "el numero marcado", "fuera del área",
    "fuera del area", "apagado o fuera", "temporalmente fuera",
    "servicio de mensajes", "servicio de voz",
    # Asistentes de voz virtuales (Google, Alexa, etc.)
    "asistente de voz", "asistente virtual", "estoy usando un asistente",
    "robot", "automatizado", "sistema automatizado",
    "inteligencia artificial", "asistente personal",
    # Ingles (por si acaso)
    "voicemail", "leave a message", "after the tone", "beep",
    "not available", "please call back", "voice assistant"
]

# Puerto del servidor
SERVER_HOST = os.getenv("AMD_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("AMD_PORT", "8765"))
