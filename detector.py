import time
from unittest import result
import numpy as np
import webrtcvad as wtcvad
import aubio as aubio

# import para websocket audio
from fastapi import FastAPI, WebSocket
import json

from faster_whisper import WhisperModel
import scipy.signal

from starlette.websockets import WebSocketDisconnect

import uvicorn

# para guardar archivos como wav
import wave


WHISPER_MODEL = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
    )

MACHINE_KEYWORDS = [
    "casilla de voz",
    "deje su mensaje",
    "despues de la señal",
    "presione cualquier tecla",
    "buzon",
    "mensaje",
    "tono",
]

HUMAN_KEYWORDS = [
        "alo",
        "hola",
        "diga",
        "bueno",
        "si diga",
]

app = FastAPI()

# SAMPLE RATE
SAMPLE_RATE_DEFAULT = 8000

# LIMITE DE ALMACENAMIENTO DE AUDIO EN BUFFER
LIMIT_BUFFER_MS = 1500

# 16000 bytes = 1 segundo = 1000 ms
# 1500 ms * 8000 bytes * 2 (int16) / 1000 ms = 24000 bytes
LIMIT_BUFFER_BYTES = int(
    SAMPLE_RATE_DEFAULT * 2 * (LIMIT_BUFFER_MS / 1000)
)

# ARRAY DE FRECUENCIAS EN LAS QUE DETECTAMOS EL BUZON
BUZON_FREQS_HZ = [350.0, 440.0, 480.0, 620.0, 950.0, 1400.0, 1800.0]

# TOLERANCIA DE FRECUENCIA PARA DETECTAR EL BUZON (EJM. ±30Hz)
BUZON_FREQ_TOLERANCE_HZ = 30.0

# LIMITE DE PUNTAJE PARA DECIDIR SI ES HUMANO
HUMAN_THRESHOLD = 3.0
# LIMITE DE PUNTAJE PARA DECIDIR SI ES BUZON
BUZON_THRESHOLD = 2.0

# VARIABLE DE AGRESIVIDAD DE WEBRTCVAD
VAR_AGRESSIVENESS = 2


def score_numpy (numpy_rms: float) -> tuple [float, float]:
    if numpy_rms is None:
        return (0.0, 0.0)
    
    if numpy_rms < 0.005:
        return (-0.2, 0.0)
    elif (numpy_rms >= 0.005) and (numpy_rms < 0.01):
        return (0.0, 0.5)
    else:
        return (0.5, -0.1)
    
def score_goertzel (goertzel_score: float) -> tuple [float, float]:
    if goertzel_score is None:
        return (0.0, 0.0)
    
    if goertzel_score < 0.3:
        return (0.1, -0.2)
    elif (goertzel_score >= 0.3) and (goertzel_score < 0.5):
        return (-0.1, 0.3)
    elif (goertzel_score >= 0.5) and (goertzel_score < 0.7):
        return (-0.2, 0.6)
    else:
        return (-0.3, 1.0)
    

def score_webrtcvad (webrtcvad_score: float) -> tuple [float, float]:
    if webrtcvad_score is None:
        return (0.0, 0.0)
    
    if webrtcvad_score < 0.2:
        return (0.0, 0.5)
    elif (webrtcvad_score >= 0.2) and (webrtcvad_score < 0.5):
        return (0.2, -0.1)
    elif (webrtcvad_score >= 0.5) and (webrtcvad_score < 0.8):
        return (0.3, -0.2)
    else:
        return (0.5, -0.3)
    

def score_f0_pitch (f0_pitch_score: float) -> tuple [float, float]:
    if f0_pitch_score is None:
        return (0.0, 0.0)

    if f0_pitch_score < 80:
        return (0.0, 0.0)
    elif (f0_pitch_score >= 80) and (f0_pitch_score < 300):
        return (0.7, -0.4)
    elif (f0_pitch_score >= 300) and (f0_pitch_score < 900):
        return (0.1, 0.1)
    elif (f0_pitch_score >= 900) and (f0_pitch_score < 1600):
        return (-0.3, 0.7)
    else:
        return (0.0, 0.2)

# Funcion para Goertzel
def gc(samples, sample_rate, target_freq):

    n = len(samples)

    if n == 0:
        return 0.0

    k = int(0.5 + ((n * target_freq) / sample_rate))

    omega = (2.0 * np.pi * k) / n

    coeff = 2.0 * np.cos(omega)

    q0 = 0.0
    q1 = 0.0
    q2 = 0.0

    for sample in samples:
        q0 = coeff * q1 - q2 + sample
        q2 = q1
        q1 = q0

    power = q1**2 + q2**2 - coeff * q1 * q2

    return float(power)

@app.websocket("/ws/amd-cascade/{call_id}")
async def amd_cascada_websocket(websocket: WebSocket, call_id: str):
    
    await websocket.accept()

    print(f"llamada conectada: {call_id}")

    
    # PRMERO RECIBE JSON DE METADATA    
    meta = await websocket.receive_json()
    print(f"metadata recibida: {meta}")

    cascada = CascadaAMDClass()


    while True:

        chunk = await websocket.receive_bytes()

        cascada.ring_buffer.extend(chunk)

        # Acumulamos en el buffer y lo limitamos a un tamaño maximo (ejm. 1500 ms)
        if len(cascada.ring_buffer) > LIMIT_BUFFER_BYTES:
            # Falback para llamar a Whisper
            cascada.ring_buffer = cascada.ring_buffer[-LIMIT_BUFFER_BYTES:]
        
        # Jalamos funcion de analisis para una llamada
        result = cascada.analize_audio()

        print(result)
        
        # Pasamos resultados cada que se actualiza el buffer
        await websocket.send_json({"type": "parcial", ** result })


        # Si ya tenemos la decision de buzon o humano, llamamos
        if result["desiciion"] in ["humano", "buzón"]:
            await websocket.send_json({"type":"final",
            "source": "dsp",
            ** result })

            print("Decision tomada, terminando analisis de audio.")
            await websocket.close()
            break
        
        # Si no tenemos la certeza, pasamos a Whisper para que transcriba y clasifique el audio acumulado en el buffer
        if len(cascada.ring_buffer) >= LIMIT_BUFFER_BYTES:
            print("Fallback a Whisper por buffer lleno sin decision clara.")

            whisper_result = cascada.detect_whisper()

            await websocket.send_json({"type":"final",
                "source": "whisper",
            ** whisper_result })

            await websocket.close()
            break

        print(f"Chunk recibido de: {len(chunk)}")

# Definimos pesos por modelos primero
weight_numpy = 0.15
weight_goertzel = 0.30
weight_webrtcvad = 0.25
weight_f0_pitch = 0.30



# Ventana de ms por modelo
Numpy_window_ms = 40
Goertzel_window_ms = 100
VAD_window_ms = 100
F0_window_ms = 100
whisper_window_ms = 1500

# Convertimos ms a bytes para cada modelo
numpy_window_bytes = int(SAMPLE_RATE_DEFAULT * 2 * (Numpy_window_ms / 1000))
goertzel_window_bytes = int(SAMPLE_RATE_DEFAULT * 2 * (Goertzel_window_ms / 1000))
vad_window_bytes = int(SAMPLE_RATE_DEFAULT * 2 * (VAD_window_ms / 1000))
f0_window_bytes = int(SAMPLE_RATE_DEFAULT * 2 * (F0_window_ms / 1000))
whisper_window_bytes = int(SAMPLE_RATE_DEFAULT * 2 * (whisper_window_ms / 1000))


class CascadaAMDClass:
    def __init__(self, sample_rate: int=8000):

        self.sample_rate = sample_rate
        
        # Declaramos y tomamos decision si llega a umnbral    
        self.decision  = None

        # Inicializamos scores
        self.score_human = 0.0
        self.score_buzon = 0.0

        # Definimos el buffer
        self.ring_buffer = bytearray()

        # Inicializamos VAD de WebRTC con el nivel de agresividad definido arriba
        self.vad = wtcvad.Vad(VAR_AGRESSIVENESS)


        # Inicializamos el detector de f0/pitch de Aubio con los parametros definidos arriba
        self.pitch_detector = aubio.pitch("yin", 1024, 800, SAMPLE_RATE_DEFAULT)

        self.pitch_detector.set_unit("Hz")
        self.pitch_detector.set_silence(-40)


    # Devuelve energia de la ventana de audio usando numpy
    def detect_numpy(self, audio_window: bytes) -> float:
        
        # PCM (codec) int16 -> float32 normalizado
        samples = np.frombuffer(audio_window, dtype=np.int16)
        
        # Para evitar ventanas vacias
        if len(samples) == 0:
            return 0.0
        
        # Normalizamos
        samples32 = samples.astype(np.float32) / 32768.0

        # Calculamos RMS
        rms = np.sqrt(np.mean(samples32**2))

        return float(rms)


    # Devuelve energia en las frecuencias definidas usando Goertzel (sacamos el ratio)
    def detect_goertzel(self, audio_window: bytes, sample_rate: int):

        # PCM (codec) int16 -> float32 normalizado
        samples = np.frombuffer(audio_window, dtype=np.int16)

        if len(samples) == 0:
            return 0.0

        samples_f32 = samples.astype(np.float32) / 32768.0
        
        # Energia total en frecuencias objetivo (declaracion)
        energy_target = 0.0

        for freq in BUZON_FREQS_HZ:
            power = gc(samples_f32, sample_rate, freq)

            # Suma de energias de todas las frecuencias objetivo (buzon)
            energy_target += power

        # Energia total de la ventana 
        total_energy = np.sum(samples_f32 ** 2)

        # Evitamos divisiones por cero
        if total_energy <= 1e-12:
            return 0.0

        # Ratio de energia en frecuencias objetivo vs energia total
        ratio = energy_target / total_energy    

        return float(ratio)



    # Devuelve energia en las frecuencias definidas usando Goertzel (sacamos el ratio)
    def detect_webrtcvad(self, audio_window : bytes, vad, sample_rate, frame=20):

        # PCM (codec) int16 -> float32 normalizado
        sample_i16 = np.frombuffer(audio_window, dtype=np.int16)

        # Conversion a bytes para procesar con webrtcvad
        audio_bytes = sample_i16.tobytes()

        # Calculamos el tamaño de cada frame en bytes (ejm. 20ms de audio que son 320 bytes)
        frame_bytes = int(sample_rate * frame / 1000) * 2 

        # Evtamos procesar ventanas menores al tamaño del frame (ej. 20ms)
        if len(audio_bytes) < frame_bytes:
            return 0.0
        
        # Contadores de frames de voz
        speech_frames = 0

        # Contador de frames totales
        total_frames = 0

        # Bucle para procesar cada frame de audio
        for i in range(0, len(audio_bytes)-frame_bytes + 1, frame_bytes):
            
            # Extraemos el frame actual
            frame = audio_bytes[i:i+frame_bytes]
            
            try:
                
                # Ejecutamos el VAD de WebRTC en el frame actual
                if vad.is_speech(frame, sample_rate):
                    # Si el frame es detectado como voz, incrementamos 1
                    speech_frames += 1

            except Exception:
                pass
            
            total_frames += 1

        # Evitamos darle score a ventanas sin frames procesados
        if total_frames == 0:
            return 0.0

        # Score de probabilidad
        score = speech_frames / total_frames

        return float(score)


    # Devuelve la frecuencia fundamental (F0) de la ventana de audio usando Aubio
    def detect_f0_pitch(self,  audio_windows: bytes):

        # PCM (codec) int16 -> float32 normalizado
        samples = np.frombuffer(audio_windows, dtype=np.int16)

        if len(samples) == 0:
            return 0.0
        
        # Normalizamos
        samples32 = samples.astype(np.float32) / 32768.0

        # Calculamos Pitch usando aubio
        pitch = float(self.pitch_detector(samples32)[0])

        return pitch



    #--------------------------------------------------------
    # Aqui empieza la gamificacion de scores, convertimos las detecciones en scores de humano y buzón
    #--------------------------------------------------------

    # Funcion principal para junte de analisis de audio y gamificacion de scores
    def analize_audio (self):

        print(f"Analizando buffer de tamaño: {len(self.ring_buffer)}")


        #--------------------------------------------------------
        # Seteamos ventanas de audio para cada algoritmo
        #--------------------------------------------------------

        # Ultimos 40ms de audio a 8kHz
        numpy_window = self.ring_buffer[-numpy_window_bytes:]

        goertzel_window = self.ring_buffer[-goertzel_window_bytes:]  # Ultimos 100ms de audio a 8kHz

        # Ultimos 100ms de audio a 8kHz
        vad_window = self.ring_buffer[-vad_window_bytes:]

        f0_window = self.ring_buffer[-f0_window_bytes:]  # Ultimos 100ms de audio a 8kHz


        #--------------------------------------------------------
        # Aqui empieza la cascada de deteccion
        #--------------------------------------------------------

        # Detectamos Primero con Numpy
        if len(numpy_window) < numpy_window_bytes:
            numpy = None
        else:
            with wave.open("numpy _window.raw", "wb") as wav_fle:
                    wav_fle.setnchannels(1)  # que sea mono
                    wav_fle.setsampwidth(2)  # int16 : osea 2 bytes
                    wav_fle.setframerate(SAMPLE_RATE_DEFAULT)  # 8000 Hz
                    wav_fle.writeframes(numpy_window)  # escribimos los bytes de audio de la ventana de numpy
            
            #imprimimos en logs por formula cuantos ms fueron
            duration_ms = (len(numpy_window) / (8000 * 2)) * 1000
            print(f"Duración de ventana numpy: {duration_ms} ms")
            
            # Ejecutamos Numpy
            numpy = self.detect_numpy(numpy_window)

        # Detectamos luego con Goertzel
        if len(goertzel_window) < goertzel_window_bytes:
            goertzel = None
        else:
            goertzel = self.detect_goertzel(goertzel_window, SAMPLE_RATE_DEFAULT)

        # Detectamos luego con WebRTC VAD
        if len(vad_window) < vad_window_bytes:
            webrtcvad_score = None
        else:
            webrtcvad_score = self.detect_webrtcvad(vad_window, self.vad, SAMPLE_RATE_DEFAULT)

        # Detectamos luego con F0 Pitch
        if len(f0_window) < f0_window_bytes:
            f0_pitch = None
        else:
            f0_pitch = self.detect_f0_pitch(f0_window)


        #--------------------------------------------------------
        # Convertimos las detecciones en scores de humano y buzón
        #--------------------------------------------------------

        # Score de humano y buzón usando Numpy
        dh_numpy, db_numpy = score_numpy(numpy)

        # Score de humano y buzón usando Goertzel
        dh_goertzel, db_goertzel = score_goertzel(goertzel)

        # Score de humano y buzón usando WebRTC VAD
        dh_webrtcvad, db_webrtcvad = score_webrtcvad(webrtcvad_score)

        # Score de humano y buzón usando F0 Pitch
        dh_f0_pitch, db_f0_pitch = score_f0_pitch(f0_pitch)
        

        #--------------------------------------------------------
        # Acumulamos Gamificando los scores
        #--------------------------------------------------------

        # Ponderado los scores para human
        self.score_human += (
            dh_numpy * weight_numpy + 
            dh_goertzel * weight_goertzel + 
            dh_webrtcvad * weight_webrtcvad + 
            dh_f0_pitch * weight_f0_pitch
        )

        # Ponderado los scores para buzón
        self.score_buzon += (
            db_numpy * weight_numpy + 
            db_goertzel * weight_goertzel + 
            db_webrtcvad * weight_webrtcvad + 
            db_f0_pitch * weight_f0_pitch
        )


        if self.score_human >= HUMAN_THRESHOLD:
            self.decision = "humano"
        elif self.score_buzon >= BUZON_THRESHOLD:
            self.decision = "buzón"

        return {
            # Resultados de la cascada
            "desiciion": self.decision,
            "score_human": self.score_human,
            "score_buzon": self.score_buzon,

            # Resultados de los modelos
            "numpy": numpy,
            "goertzel": goertzel,
            "webrtcvad": webrtcvad_score,
            "f0_pitch": f0_pitch,

            # Resultados de Humano y Buzon por modelo
            "db_numpy": db_numpy,
            "dh_numpy": dh_numpy,
            "db_goertzel": db_goertzel,
            "dh_goertzel": dh_goertzel,
            "db_webrtcvad": db_webrtcvad,
            "dh_webrtcvad": dh_webrtcvad,
            "db_f0_pitch": db_f0_pitch,
            "dh_f0_pitch": dh_f0_pitch,
        }



    # Primero trasncribimos con Whisper
    def transcribe_with_whisper(self):
        samples = np.frombuffer(self.ring_buffer, dtype=np.int16)

        # Si no hay muestras, no se puede transcribir
        if len(samples) == 0:
            print("No hay muestras para transcribir con Whisper.")
            return None
            
        
        # Normalizamos las muestras a float32
        audio_f32 = samples.astype(np.float32) / 32768.0

        # Resampleamos a 16000 Hz si es necesario
        audio_16k = scipy.signal.resample_poly(audio_f32, 16000,self.sample_rate).astype(np.float32)

        segments , info = WHISPER_MODEL.transcribe(    audio_16k,
            language="es",
            beam_size=1,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False
        )

        text = " ".join([segment.text for segment in segments]).lower().strip()

        return text

    # Clasificamos la transcripcion de Whisper buscando keywords de buzon y humano
    def classify_with_whisper(self, text: str) -> dict:
        if not text:
            return {"decision": "desconocido", "is_buzon": "transcripcion vacia"}

        # Iteramos para buscar las keywods de buzon primero
        for keyword in MACHINE_KEYWORDS:

            if keyword in text:
                return {"decision": "buzón", "reason": f"se detecto la palabra clave '{keyword}' en la transcripcion",
                        "transcripcion": text}
        
        # Iteramos para buscar las keywods de humano luego
        for keyword in HUMAN_KEYWORDS:

            if keyword in text:
                return {"decision": "humano",
                        "reason": f"se detecto la palabra clave '{keyword}' en la transcripcion",
                        "transcripcion": text}
        
        return {
            "decision": "desconocido",
            "reason": "no se detectaron palabras clave de buzon ni de humano en la transcripcion",
            "transcription": text
        }
        
    # Tomamos la decision final de Whisper
    def detect_whisper (self):
        text =  self.transcribe_with_whisper()

        result = self.classify_with_whisper(text)

        return result
    


if __name__ == "__main__":
    print("Iniciando servidor de AMD en WebSocket...")

    uvicorn.run(app, host="0.0.0.0",
                port=8765)