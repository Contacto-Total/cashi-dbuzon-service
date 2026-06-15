import time
from unittest import result
import numpy as np
import webrtcvad as wtcvad
import aubio as aubio

# import para websocket audio
from fastapi import FastAPI, WebSocket
import json

# import Whisper con Faster
from faster_whisper import WhisperModel
import scipy.signal

from starlette.websockets import WebSocketDisconnect
import uvicorn

# para guardar archivos como wav
import wave

# Liberias para auditoria de hora
from datetime import datetime
import time
import unicodedata
import re

from dotenv import load_dotenv
import os

# Import para GPT MINI TRANSCRIBE
from openai import AsyncOpenAI
import io
import soundfile as sf

import asyncio
from concurrent.futures import ThreadPoolExecutor

from amd_lr import StreamingAMD, AMDConfig, Label, Decision

WHISPER_POOL = ThreadPoolExecutor(max_workers=3)

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=10.0,
    max_retries=1)





WHISPER_MODEL = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    )

# Keywords de buzon con variantes de typos de whisper-tiny
MACHINE_KEYWORDS = [
      "de voz", "comunicado con", "te has comunicado",
      "casilla", "cacilla", "cassiye", "casiya", "cacilya", "castilla", "cocilla",
      "transferida", "transferido", "transcedida", "torncerida",
      "llamada sera", "llamada serÃ¡", "llamada fera", "llamada cera",
      "mensaje",
      "buzon", "buzÃ³n", "busÃ³n", "buson",
      "deja tu", "dÃ©jate", "dejate", "dejalmente", "dÃ©jame",
      "deje", "grabar", "grabe",
      "tono", "seÃ±al", "senal",
      "tecla", "decla", "decala", "decada", "dÃ©cada", "dÃ©cala", "dÃ©cila", "trecle",
      "presione", "precione", "preciÃ³", "precio", "prefiÃ³", "prefiero",
      "cualquier", "terminar",
      "despues del", "despuÃ©s del", "despues de", "despuÃ©s de",
      "no se encuentra",
      "no esta disponible", "no estÃ¡ disponible", "no disponible",
      "numero marcado", "nÃºmero marcado",
      "desvio", "desvÃ­o", "contestador", "apagado",
      "fuera de", "cobertura", "no atiende", "ahora no puede",
]

# Buzones suelen dictar el numero llamado (6+ digitos seguidos)
DIGIT_SEQUENCE = re.compile(r"\d{6,}")

# Keywords humanas DECISIVAS (saludo corto = humano de verdad)
HUMAN_DECISIVE = [
      "alo", "alÃ³", "aloo", "halo", "allo", "alou", "aroh", "aro",
      "diga", "dÃ­game", "digame", "diga me",
      "bueno", "buano", "weno",
      "buenos dÃ­as", "buenos dias", "buenos dia", "weno dia", "buen dia",
      "buenas tardes", "buena tarde", "buenas noches", "buena noche", "buenas",
      "si diga", "sÃ­ diga", "sÃ­ dÃ­game", "si digame",
      "quien habla", "quiÃ©n habla", "con quien", "con quiÃ©n", "con quiÃ©n hablo",
]

# Keywords humanas AMBIGUAS (buzones tambien dicen "hola")
HUMAN_AMBIGUOUS = [
      "hola","ola","ohla",
]

app = FastAPI()

# SAMPLE RATE
SAMPLE_RATE_DEFAULT = 8000

# LIMITE DE ALMACENAMIENTO DE AUDIO EN BUFFER
LIMIT_BUFFER_MS = 1800

# 16000 bytes = 1 segundo = 1000 ms
# 1500 ms * 8000 bytes * 2 (int16) / 1000 ms = 24000 bytes
LIMIT_BUFFER_BYTES = int(
    SAMPLE_RATE_DEFAULT * 2 * (LIMIT_BUFFER_MS / 1000)
)

# ARRAY DE FRECUENCIAS EN LAS QUE DETECTAMOS EL BUZON
BUZON_FREQS_HZ = [350.0, 440.0, 480.0, 620.0, 950.0, 1400.0, 1800.0]

# TOLERANCIA DE FRECUENCIA PARA DETECTAR EL BUZON (EJM. Â±30Hz)
BUZON_FREQ_TOLERANCE_HZ = 30.0

# LIMITE DE PUNTAJE PARA DECIDIR SI ES HUMANO
HUMAN_THRESHOLD = 0.98
# LIMITE DE PUNTAJE PARA DECIDIR SI ES BUZON
BUZON_THRESHOLD = 0.95

# VARIABLE DE AGRESIVIDAD DE WEBRTCVAD
VAR_AGRESSIVENESS = 2


def score_numpy (numpy_rms: float) -> tuple [float, float]:
    if numpy_rms is None:
        return (0.0, 0.0)
    
    if numpy_rms > 0.06:
        return (-0.0326666666666667, 0.076)
    elif (numpy_rms > 0.02):
        return (0.049, -0.038)
    else:
        return (0.0, 0.0)
    
def score_goertzel (goertzel_score: float) -> tuple [float, float]:
    if goertzel_score is None:
        return (0.0, 0.0)
    
    if goertzel_score > 0.45:
        return (-0.0653333333333333, 0.152)
    else:
        return (0.0, 0.0)
    
def score_webrtcvad (webrtcvad_score: float) -> tuple [float, float]:
    if webrtcvad_score is None:
        return (0.0, 0.0)
    
    if webrtcvad_score < 0.3:
        return (0.147, -0.114)
    elif (webrtcvad_score >= 0.3) and (webrtcvad_score < 0.55):
        return (0.0653333333333333, -0.057)
    elif (webrtcvad_score >= 0.55) and (webrtcvad_score <= 0.75):
        return (0.0, 0.0)
    else:
        return (-0.0816666666666667, 0.171)

# -------------------------------------------------------
# SCORING DE PITCH POR VENTANA DE 100 MS SIN DESVIACION 
# -------------------------------------------------------
def score_f0_pitch (f0_std: float, f0_avg: float, f0_n: int) -> tuple [float, float]:
      # agudo (media estable) -> humano
      if f0_avg is not None and f0_n >= 2 and f0_avg > 250:
          return (0.196, -0.152)
      # expresivo -> humano
      if f0_std is not None and f0_std > 53:
          return (0.130666666666667, -0.095)
      # monotono -> humano (gateado por muestras: usa el acumulado ya estable)
      if f0_std is not None and f0_n >= 15 and f0_std < 9.5:
          return (0.163333333333333, -0.114)
      return (0.0, 0.0)
# -------------------------------------------------------
# SCORING DE PITCH POR VENTANA DE 500 MS CONs DESVIACION 
# -------------------------------------------------------
"""
def score_f0_pitch (f0_pitch_score: float) -> tuple [float, float]:
    if f0_pitch_score is None:
        return (0.0, 0.0)

    if f0_pitch_score < 40:
        return (-0.05, 0.1)
    elif (f0_pitch_score >= 40) and (f0_pitch_score < 70):
        return (0.0, 0.05)
    elif (f0_pitch_score >= 70) and (f0_pitch_score < 100):
        return (0.0, 0.0)
    elif (f0_pitch_score >= 100) and (f0_pitch_score < 150):
        return (0.1, -0.05)
    else:
        return (0.15, -0.1)
    """


def score_human(rms_max, rms_avg, vad_ratio, f0_avg, rms_count, score_buzon, score_human):
    if rms_count is None or rms_count < 45:
        return (0.0,0.0)
    # Gate de energia: Cuando humano se queda callado pero hay ruido
    if rms_max < 0.0455:
        return (0.196, -0.152)
    # Gate de pausas largas: Cuando humano se queda callado por pausas largas
    if rms_avg is not None and rms_avg < 0.038 and rms_max > 0.22:
        return (0.196, -0.152)
    # Pitch bajo + pausas: Cuando humano se queda callado por pausas largas
    if (f0_avg is not None and vad_ratio is not None and f0_avg < 227 and vad_ratio < 0.38):
        return (0.130666666666667, -0.095)
    
    if (rms_count is None or rms_count >=70 and vad_ratio is not None and vad_ratio < 0.55
        and score_buzon < 0 and ((score_human/ 0.326667) - (score_buzon/ 0.38)) >= 2.0):
        return (0.196, -0.152)
    return (0.0, 0.0)


        
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

    print(f"\nâ–¶ llamada {call_id}")

    
    # PRMERO RECIBE JSON DE METADATA    
    meta = await websocket.receive_json()
    # print(f"metadata recibida: {meta}")

    cascada = CascadaAMDClass()

    amd = StreamingAMD()

    started_at = time.time()
    contador_chunk = 0

    decision = None
    while decision is None:
        try:
            chunk = await websocket.receive_bytes()
        except WebSocketDisconnect:
            p = amd.current_p_humano()
            decision = Decision(Label.BUZON if p <= amd.cfg.p_buzon else Label.HUMANO,
                                p, amd.elapsed_ms, forced=True)
            break
        
        # 1. Acumulamos el audio para el STT (PARA GPT)
        cascada.ring_buffer.extend(chunk)

        # 2. Recalculamos humano cada 20 ms
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        contador_chunk +=1
        decision = amd.feed_chunk(samples)
        
    # Si RL ya decidio
    elapsed_md = round((time.time() - started_at) * 1000, 2)
    event_base = {
        "ts": datetime.now().isoformat(),
        "chunk": contador_chunk,
        "elapsed": elapsed_md,
        "buffer_bytes": len(cascada.ring_buffer),
    }

    if decision.label == Label.BUZON:
        # SI es buzom
        result = {
            **event_base,
            "decision": "buzon",
            "reason": f"RL p_human={decision.p_human:3f}",
            "transcripcion": "",}
        fuente = "RL"
    else:
        # Si no es humano
        try:
            # result = await cascada.detect_()
            result = {"decision": "sindetectar", "reason": "modo prueba - paso a gpt", "transcripcion": ""}
            fuente ="gpt"
        except Exception as e:
            print(f"GPT fallo ({type(e).__name__}) -> whisper")
            try:
                result = await cascada.detect_whisper()
                fuente = "whisper"
            except Exception as e:
                print(f"WHISPER fallo ({type(e).__name__}) -> nada")
                result = {"decision": "humano", "reason": "fail-safe", "transcripcion": ""}
                fuente = "fail-safe"
    
    print(f"  {fuente} -> {result['decision'].upper()} (p_human={decision.p_human:.3f})")       

    payload = {
          "type": "decision",
          "source": fuente,
          **event_base,
          "decision": {
            "result": result["decision"],
            "reason": result.get("reason"),
            "transcription": result.get("transcripcion"),
            "decided_at_chunk": contador_chunk,
            "decided_at_ms": elapsed_md,
            "p_human": round(decision.p_human, 3),
        },
    }
    await websocket.send_json(payload)
    await websocket.close()         
                
        

# Definimos pesos por modelos primero
weight_numpy = 0.15
weight_goertzel = 0.10
weight_webrtcvad = 0.30
weight_f0_pitch = 0.45
weight_loudness = 0.23



# Ventana de ms por modelo
Numpy_window_ms = 40
Goertzel_window_ms = 60 # 100 ms
VAD_window_ms = 80 # 80 ms
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

        # Contador para wavs de numpy 40 ms
        self.counter_buffer=0

        # Historial de F0 para calcular desviacion
        self.f0_history = []

        # VARIABLES PARA ANALIZAR EL ACUMULADO DE VAD DE TODA LA LLAMADA
        self.vad_sum = 0.0
        self.vad_count = 0

        # VARIABLES PARA ANALIZAR EL ACUMULADO DE RMS DE TODA LA LLAMADA
        self.rms_sum = 0.0
        self.rms_count = 0
        self.rms_max = 0.0

        self.ah_f0 = 0.0
        self.ah_rms = 0.0

        self.f0_sum=0.0;
        self.f0_n=0




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

        # Ratio de energia en frecuencias objetivo vs energia total.
        # Normalizamos por N (Parseval: Sum|X[k]|^2 = N * Sum x[n]^2), si no el
        # "ratio" escala con N (~O(N)) y se dispara a decenas/cientos en vez de 0-1.
        n = len(samples_f32)
        ratio = energy_target / (n * total_energy)

        return float(ratio)



    # Devuelve energia en las frecuencias definidas usando Goertzel (sacamos el ratio)
    def detect_webrtcvad(self, audio_window : bytes, vad, sample_rate, frame=20):

        # PCM (codec) int16 -> float32 normalizado
        sample_i16 = np.frombuffer(audio_window, dtype=np.int16)

        # Conversion a bytes para procesar con webrtcvad
        audio_bytes = sample_i16.tobytes()

        # Calculamos el tamaÃ±o de cada frame en bytes (ejm. 20ms de audio que son 320 bytes)
        frame_bytes = int(sample_rate * frame / 1000) * 2 

        # Evtamos procesar ventanas menores al tamaÃ±o del frame (ej. 20ms)
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


    # -----------------------------------------------------
    #  CALCULO EN BASE A BUFFER DE 500 MS CON LA DESVIACION 
    # -----------------------------------------------------
    """
    def detect_f0_pitch(self,  audio_windows: bytes):
        # PCM (codec) int16 -> float32 normalizado
        samples = np.frombuffer(audio_windows, dtype=np.int16)

        if len(samples) == 0:
            return 0.0
        
        # Normalizamos
        samples32 = samples.astype(np.float32) / 32768.0

        frame=800
        step=400

        pitches = []
        for i in range(0, len(samples32)-frame + 1, step):
            p = float(self.pitch_detector(samples32[i:i+frame])[0])
            if 70 <= p <= 400:  # Filtramos pitches fuera del rango tÃ­pico de voz humana
                pitches.append(p)

        if len(pitches) < 4:
            return None
        
        return float(np.std(pitches))
    """


    # -----------------------------------------------------
    #  CALCULO EN BASE A CADA CHUNKS DE LA VENTANA DE 100 MS
    # -----------------------------------------------------
    # Devuelve la frecuencia fundamental (F0) de la ventana de audio usando Aubio
    def detect_f0_pitch(self,  audio_windows: bytes):

        # PCM (codec) int16 -> float32 normalizado
        samples = np.frombuffer(audio_windows, dtype=np.int16)

        if len(samples) == 0:
            return None
        
        # Normalizamos
        samples32 = samples.astype(np.float32) / 32768.0

        # Calculamos Pitch usando aubio
        pitch = float(self.pitch_detector(samples32)[0])

        return pitch if 70 <= pitch <= 400 else None


    #--------------------------------------------------------
    # Aqui empieza la gamificacion de scores, convertimos las detecciones en scores de humano y buzÃ³n
    #--------------------------------------------------------

    # Funcion principal para junte de analisis de audio y gamificacion de scores
    def analize_audio (self):

        # print(f"Analizando buffer de tamaÃ±o: {len(self.ring_buffer)}")


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

        """
        filename = f"buffer_window_{self.counter_buffer}.wav"
        with wave.open(filename, "wb") as wav_fle:
                    wav_fle.setnchannels(1)  # que sea mono
                    wav_fle.setsampwidth(2)  # int16 : osea 2 bytes
                    wav_fle.setframerate(SAMPLE_RATE_DEFAULT)  # 8000 Hz
                    wav_fle.writeframes(self.ring_buffer)  # escribimos los bytes de audio de la ventana de numpy
        self.counter_buffer += 1
        """

        # Detectamos Primero con Numpy
        if len(numpy_window) < numpy_window_bytes:
            numpy = None
        else:
            
            # Ejecutamos Numpy
            numpy = self.detect_numpy(numpy_window)
        r = numpy if numpy is not None else 0.0
        self.rms_sum += r
        self.rms_count += 1
        self.rms_max = max(self.rms_max, r)
        rms_avg = self.rms_sum / self.rms_count if self.rms_count >= 20 else None

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
        
        v = webrtcvad_score if webrtcvad_score is not None else 0.0
        if v > 0 or self.vad_count > 0:
            self.vad_sum += v
            self.vad_count += 1
        vad_ratio = self.vad_sum / self.vad_count if self.vad_count >= 20 else None

        # Detectamos luego con F0 Pitch
        if len(f0_window) < f0_window_bytes:
            f0_pitch = None
        else:
            f0_pitch = self.detect_f0_pitch(f0_window)
        # 
        if f0_pitch is not None:
            self.f0_history.append(f0_pitch)
            self.f0_sum += f0_pitch
            self.f0_n += 1
        # Media movil de F0 para toda la llamada (acumulado)
        f0_avg_run = self.f0_sum / self.f0_n if self.f0_n > 0 else None
        if len(self.f0_history) >= 6:  # Mantener solo las Ãºltimas 6 mediciones de F0
            f0_std = float(np.std(self.f0_history))
        else: 
            f0_std = None


        #--------------------------------------------------------
        # Convertimos las detecciones en scores de humano y buzÃ³n
        #--------------------------------------------------------

        # Score de humano y buzÃ³n usando Numpy
        dh_numpy, db_numpy = score_numpy(rms_avg)
        self.ah_rms += dh_numpy

        # Score de humano y buzÃ³n usando Goertzel
        dh_goertzel, db_goertzel = score_goertzel(goertzel)

        # Score de humano y buzÃ³n usando WebRTC VAD
        dh_webrtcvad, db_webrtcvad = score_webrtcvad(vad_ratio)
        # Gate de energia: sin energia, el VAD no es confiable -> no vota
        if rms_avg is not None and rms_avg < 0.005:
            dh_webrtcvad, db_webrtcvad = 0.0, 0.0

        # Score de humano y buzÃ³n usando F0 Pitch
        dh_f0_pitch, db_f0_pitch = score_f0_pitch(f0_std, f0_avg_run, self.f0_n)
        self.ah_f0 += dh_f0_pitch
        
        # Score en base a analisis de patron de humano
        dh_loud, db_loud = score_human(self.rms_max, rms_avg, vad_ratio, f0_avg_run, self.rms_count, self.score_buzon, self.score_human)
        self.score_human += dh_loud * weight_loudness
        self.score_buzon += db_loud * weight_loudness

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

        # Ponderado los scores para buzÃ³n
        self.score_buzon += (
            db_numpy * weight_numpy + 
            db_goertzel * weight_goertzel + 
            db_webrtcvad * weight_webrtcvad + 
            db_f0_pitch * weight_f0_pitch
        )

        if self.score_human >= HUMAN_THRESHOLD:
            if (f0_avg_run is not None and f0_avg_run < 163 and rms_avg is not None and rms_avg > 0.04):
                self.decision = None
            else:
                self.decision = "humano"
        elif self.score_buzon >= BUZON_THRESHOLD:
            if rms_avg is not None and rms_avg < 0.005:
                self.decision = None
            elif self.score_buzon < 0.98 and  (self.ah_f0 >= 2.61 or self.ah_rms >= 3.27):
                self.decision = None
            else:
                self.decision = "buzon"

    
        return {
            # Resultados de la cascada
            "decision": self.decision,

            "scores": {
                "human": self.score_human,
                "buzon": self.score_buzon
            },

            "models": {
                # Resultados de los modelos
                "rms": {
                    "value": numpy
                },
                "goertzel": {
                    "value": goertzel
                },
                "vad": {
                    "value": webrtcvad_score
                },
                "f0": {
                    "value": f0_pitch
                }
            },
            # Resultados de Humano y Buzon por modelo
            "contrib_human": {
                "rms": dh_numpy,
                "goertzel": dh_goertzel,
                "vad": dh_webrtcvad,
                "f0": dh_f0_pitch
            },

            "contrib_buzon": {
                "rms": db_numpy,
                "goertzel": db_goertzel,
                "vad": db_webrtcvad,
                "f0": db_f0_pitch
            }
    }



    # TRANSCRIPCION CON GPT-o4 MINI TRANSCRIBE
    async def transcribe_whi_gtp4_mini(self):
        samples = np.frombuffer(self.ring_buffer, dtype=np.int16)

        # Si no hay muestras, no se puede transcribir
        if len(samples) == 0:
            print("No hay muestras para transcribir con Whisper.")
            return None
        
        # Normalizamos las muestras a float32
        audio_f32 = samples.astype(np.float32) / 32768.0

        # Resampleamos a 16000 Hz si es necesario
        audio_16k = scipy.signal.resample_poly(audio_f32, 16000,self.sample_rate).astype(np.float32)

        # Normalizamos el volumen: el audio telefonico es bajo y whisper espera audio normalizado.
        # Subimos el pico a ~0.95 para que tiny escuche claro (mejora la transcripcion sin costo).
        peak = np.max(np.abs(audio_16k))
        if peak > 0:
            audio_16k = (audio_16k * (0.95 / peak)).astype(np.float32)

        wav_buffer = io.BytesIO()
        
        sf.write(wav_buffer, audio_16k, 16000, format="WAV")

        wav_buffer.seek(0)

        transcript = await client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=("audio.wav",wav_buffer, "audio/wav"),
            language="es",
            prompt=(
                "Transcribe en espaÃ±ol de PerÃº. NO traduzcas al inglÃ©s. "
                "Es una llamada telefÃ³nica. Frases tÃ­picas: buzÃ³n de voz, "
                "casilla, deje su mensaje despuÃ©s del tono, presione una tecla, "
                "el nÃºmero que marcÃ³ no estÃ¡ disponible, alÃ³, hola, diga, buenas."
            ),
            temperature=0
        )

        text = transcript.text.lower().strip()
        if hasattr(transcript, "usage") and transcript.usage:
            u = transcript.usage
            print(f" tokens consumidos: {transcript.usage.total_tokens}")
        print(f" GPT text: '{text}'")

        return text


    # ------------------------------------------------------------------------------------------
    #  COMENTADO DE MOMENTO POR GPT
    # ------------------------------------------------------------------------------------------
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

        # Normalizamos el volumen: el audio telefonico es bajo y whisper espera audio normalizado.
        # Subimos el pico a ~0.95 para que tiny escuche claro (mejora la transcripcion sin costo).
        peak = np.max(np.abs(audio_16k))
        if peak > 0:
            audio_16k = (audio_16k * (0.95 / peak)).astype(np.float32)

        segments, info = WHISPER_MODEL.transcribe(audio_16k, language="es", beam_size=5,
                                            no_speech_threshold=1.0, vad_filter=False,
                                            condition_on_previous_text=False,
                                            initial_prompt=("Buzon de voz casilla mensaje tono tecla "
                                                            "transferida disponible alo hola diga"))
        
        
        seg_list = list(segments)                                   # <-- consume UNA vez
        text = " ".join([s.text for s in seg_list]).lower().strip() # <-- usa seg_list, NO segments
        print(f"  WHISPER txt: '{text}'")

        return text


    def _norm(self, t):
      # quita tildes y pasa a minÃºsculas: "busÃ³n" -> "buson"
      return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()

    # Clasificamos la transcripcion de Whisper buscando keywords de buzon y humano
    def classify_with_whisper(self, text: str) -> dict:
        if not text:
          return {"decision": "buzon", "reason": "sin transcripcion (audio no claro)", "transcripcion": ""}
        
        t = text.lower()
        machine_hits  = [kw for kw in MACHINE_KEYWORDS if kw in t]
        decisive_hits = [kw for kw in HUMAN_DECISIVE  if kw in t]
        ambiguous_hits = [kw for kw in HUMAN_AMBIGUOUS if kw in t]
        word_count = len(t.split())
        has_number = bool(DIGIT_SEQUENCE.search(t))

        # 1. Keyword explicita de buzon
        if machine_hits:
            return {"decision": "buzon", "reason": f"buzon kw={machine_hits}", "transcripcion": text}

        # 2. Buzon dictando numero de telefono
        if has_number:
            return {"decision": "buzon", "reason": "buzon dictando numero", "transcripcion": text}

        # 3. Keyword humana decisiva: corto=humano, largo=buzon que esquivo keywords
        if decisive_hits:
            if word_count <= 4:
                return {"decision": "humano", "reason": f"humano decisivo {decisive_hits} ({word_count}pal)", "transcripcion": text}
            return {"decision": "buzon", "reason": f"saludo largo con kw humana ({word_count}pal)", "transcripcion": text}

        # 4. Keyword ambigua ("hola"): corto=humano, largo=buzon
        if ambiguous_hits:
            if word_count <= 3:
                return {"decision": "humano", "reason": f"hola ambiguo ({word_count}pal)", "transcripcion": text}
            return {"decision": "buzon", "reason": f"hola en texto largo ({word_count}pal)", "transcripcion": text}

        # 5. No reconocio nada de la trasncripcion
        return {"decision": "duda", "reason": f"Sin text0 reconocible ({word_count}pal)", "transcripcion": text}

    async def classify_with_gpt (self, text: str) -> dict:
        """ gpt o4 mini para la decsion de audio que quedo en duda luego de la comparacion de keywords"""
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content":
                  "Clasificas el inicio de una llamada telefonica en Peru. Responde UNA "
                  "sola palabra, sin puntuacion: 'humano' si contesto una persona real "
                  "(alo, hola, diga, si, buenas, quien habla), o 'buzon' si es un "
                  "contestador o mensaje grabado (deje su mensaje, despues del tono, "
                  "el numero no esta disponible, casilla de voz)."},
              {"role": "user", "content": f'Transcripcion: "{text}"'},
            ],
        )
        ans = completion.choices[0].message.content.lower().strip()
        print(f" GPT decision: '{ans}'")
        if "buzon" in ans or "buzÃ³n" in ans:
            return {"decision": "buzon", "reason": "gpt o4 mini", "transcripcion": text}
        if "humano" in ans or "human" in ans:
            return {"decision": "humano", "reason": "gpt o4 mini", "transcripcion": text}
        return {"decision": "duda", "reason": f"gpt indeciso({ans})", "transcripcion": text}

    async def cascada_classify (self, text: str) -> dict:
        result = self.classify_with_whisper(text)
        
        # Primero clasificador de Python
        if result["decision"] in ("buzon", "humano"):
            return result
        

        # Segundo clasificador de GPT
        try:
            result = await self.classify_with_gpt(text)
            if result["decision"] in ("buzon", "humano"):
                return result
        except Exception as e:
            print(f"GPT clasificÃ³ mal: ({type(e).__name__})")

        return {"decision": "humano", "reason": "fallback de clasficador", "transcripcion": text}

    # Tomamos la decision con GPT
    async def detect_ (self):
        text =  await self.transcribe_whi_gtp4_mini()
        return await self.cascada_classify(text)
    
    # FALLBACK DE WHISPER SI GPT NO FUNCIONA
    async def detect_whisper (self):
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(WHISPER_POOL, self.transcribe_with_whisper)
        return await self.cascada_classify(text)

if __name__ == "__main__":
    print("Iniciando servidor de AMD en WebSocket...")
    uvicorn.run(app, host="0.0.0.0",
                port=8765)