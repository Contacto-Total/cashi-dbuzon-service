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

# Liberias para auditoria de hora
from datetime import datetime
import time


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
LIMIT_BUFFER_MS = 2000

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
BUZON_THRESHOLD = 2.5

# VARIABLE DE AGRESIVIDAD DE WEBRTCVAD
VAR_AGRESSIVENESS = 2


def score_numpy (numpy_rms: float) -> tuple [float, float]:
    if numpy_rms is None:
        return (0.0, 0.0)
    
    if numpy_rms > 0.06:
        return (-0.1, 0.2)
    elif (numpy_rms > 0.02):
        return (0.15, -0.1)
    else:
        return (0.0, 0.0)
    
def score_goertzel (goertzel_score: float) -> tuple [float, float]:
    if goertzel_score is None:
        return (0.0, 0.0)
    
    if goertzel_score > 0.45:
        return (-0.2, 0.4)
    else:
        return (0.0, 0.0)
    
def score_webrtcvad (webrtcvad_score: float) -> tuple [float, float]:
    if webrtcvad_score is None:
        return (0.0, 0.0)
    
    if webrtcvad_score < 0.3:
        return (0.45, -0.3)
    elif (webrtcvad_score >= 0.3) and (webrtcvad_score < 0.55):
        return (0.2, -0.15)
    elif (webrtcvad_score >= 0.55) and (webrtcvad_score <= 0.75):
        return (0.0, 0.0)
    else:
        return (-0.25, 0.45)

# -------------------------------------------------------
# SCORING DE PITCH POR VENTANA DE 100 MS SIN DESVIACION 
# -------------------------------------------------------
def score_f0_pitch (f0_std: float, f0_avg: float, f0_n: int) -> tuple [float, float]:
      # agudo (media estable) -> humano
      if f0_avg is not None and f0_n >= 2 and f0_avg > 250:
          return (0.6, -0.4)
      # expresivo -> humano
      if f0_std is not None and f0_std > 53:
          return (0.4, -0.25)
      # monotono -> humano (gateado por muestras: usa el acumulado ya estable)
      if f0_std is not None and f0_n >= 15 and f0_std < 9.5:
          return (0.5, -0.3)
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


def score_human(rms_max, rms_avg, vad_ratio, f0_avg, rms_count):
    if rms_count is None or rms_count < 45:
        return (0.0,0.0)
    # Gate de energia: Cuando humano se queda callado pero hay ruido
    if rms_max < 0.0455:
        return (0.6, -0.4)
    # Gate de pausas largas: Cuando humano se queda callado por pausas largas
    if rms_avg is not None and rms_avg < 0.038 and rms_max > 0.22:
        return (0.6, -0.4)
    # Pitch bajo + pausas: Cuando humano se queda callado por pausas largas
    if (f0_avg is not None and vad_ratio is not None and f0_avg < 227 and vad_ratio < 0.38):
        return (0.4, -0.25)
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

    print(f"llamada conectada: {call_id}")

    
    # PRMERO RECIBE JSON DE METADATA    
    meta = await websocket.receive_json()
    print(f"metadata recibida: {meta}")

    cascada = CascadaAMDClass()

    started_at = time.time()
    contador_chunk = 0

    while True:

        chunk = await websocket.receive_bytes()

        cascada.ring_buffer.extend(chunk)

        # Acumulamos en el buffer y lo limitamos a un tamaño maximo (ejm. 1500 ms)
        if len(cascada.ring_buffer) > LIMIT_BUFFER_BYTES:
            # Falback para llamar a Whisper
            cascada.ring_buffer = cascada.ring_buffer[-LIMIT_BUFFER_BYTES:]
        
        # Jalamos funcion de analisis para una llamada
        result = cascada.analize_audio()

        contador_chunk+=1
        
        elapsed_ms = round((time.time() - started_at) * 1000, 2)

        event_ts = datetime.now().isoformat()

        # Base para timestamp de eventos
        event_base = {
            "ts": event_ts,
            "chunk": contador_chunk,
            "elapsed_ms": elapsed_ms,
            "buffer_bytes": len(cascada.ring_buffer)
        }

        # Payload para el analisis
        payload_analisis={
            "type": "analysis",
            **event_base,
            **result
        }

        payload_analisis["verdict"] = payload_analisis.pop("decision")

        print(payload_analisis)

        # Pasamos por websocket resultados
        await websocket.send_json(payload_analisis)



        # Si ya tenemos la decision de buzon o humano, llamamos
        if result["decision"] in ["humano", "buzon"]:
            payload_decision={
                "type": "decision",
                "source": "dsp",

                **event_base,

                "decision": {
                    "result": result["decision"],
                    "scores": result["scores"],
                    "models": result["models"],
                    "contrib_human": result["contrib_human"],
                    "contrib_buzon": result["contrib_buzon"],
                    "decided_at_chunk": contador_chunk,
                    "decided_at_ms": elapsed_ms
                }
            }

            print(payload_decision)
            
            await websocket.send_json(payload_decision)

            print("Desicion tomada, analisis terminado")

            await websocket.close()
            break


        # ------------------------------------------------------------------------
        # TEMPORAL
        # ------------------------------------------------------------------------
        # --- TOPE DE TIEMPO (1800ms) -> SIN DECISION (sin Whisper) ---
        # El ring_buffer se llena a LIMIT_BUFFER_BYTES (28800 = 1800ms) justo
        # en el chunk 90. Si llegamos aqui es porque NO hubo decision DSP.
        # Cerramos como "sindetectar" para medir cuantos pasarian a Whisper.
        if len(cascada.ring_buffer) >= LIMIT_BUFFER_BYTES:
            payload_sindecision = {
                "type": "decision",
                "source": "dsp_timeout",
                **event_base,
                "decision": {
                    "result": "sindetectar",
                    "scores": result["scores"],
                    "decided_at_chunk": contador_chunk,
                    "decided_at_ms": elapsed_ms
                }
            }
            print("Tope 1800ms sin decision -> sindetectar")
            await websocket.send_json(payload_sindecision)
            await websocket.close()
            break

        """

        # COMENTAOD PARA QUE NO PASE A WHISPER PARA PRUEBAS

        # Si no tenemos la certeza, pasamos a Whisper para que transcriba y clasifique el audio acumulado en el buffer
        if len(cascada.ring_buffer) >= LIMIT_BUFFER_BYTES:
            print("Fallback a Whisper por buffer lleno sin decision clara.")

            whisper_result = cascada.detect_whisper()

            payload_whisper={
                "type": "decision",
                "source": "whisper",

                **event_base,

                "decision": {
                    "result": whisper_result["decision"],
                    "reason": whisper_result.get("reason"),
                    "transcription": whisper_result.get("transcripcion"),
                    "decided_at_chunk": contador_chunk,
                    "decided_at_ms": elapsed_ms
                }
            }

            print(payload_whisper)

            await websocket.send_json(payload_whisper)

            await websocket.close()
            break
        """

        print(f"Chunk recibido de: {len(chunk)}")

# Definimos pesos por modelos primero
weight_numpy = 0.15
weight_goertzel = 0.10
weight_webrtcvad = 0.30
weight_f0_pitch = 0.45
weight_loudness = 0.20



# Ventana de ms por modelo
Numpy_window_ms = 40
Goertzel_window_ms = 60 # 100 ms
VAD_window_ms = 80 # 100 ms
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
            if 70 <= p <= 400:  # Filtramos pitches fuera del rango típico de voz humana
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
        if len(self.f0_history) >= 6:  # Mantener solo las últimas 6 mediciones de F0
            f0_std = float(np.std(self.f0_history))
        else: 
            f0_std = None


        #--------------------------------------------------------
        # Convertimos las detecciones en scores de humano y buzón
        #--------------------------------------------------------

        # Score de humano y buzón usando Numpy
        dh_numpy, db_numpy = score_numpy(rms_avg)
        self.ah_rms += dh_numpy

        # Score de humano y buzón usando Goertzel
        dh_goertzel, db_goertzel = score_goertzel(goertzel)

        # Score de humano y buzón usando WebRTC VAD
        dh_webrtcvad, db_webrtcvad = score_webrtcvad(vad_ratio)
        # Gate de energia: sin energia, el VAD no es confiable -> no vota
        if rms_avg is not None and rms_avg < 0.005:
            dh_webrtcvad, db_webrtcvad = 0.0, 0.0

        # Score de humano y buzón usando F0 Pitch
        dh_f0_pitch, db_f0_pitch = score_f0_pitch(f0_std, f0_avg_run, self.f0_n)
        self.ah_f0 += dh_f0_pitch
        
        # Score en base a analisis de patron de humano
        dh_loud, db_loud = score_human(self.rms_max, rms_avg, vad_ratio, f0_avg_run, self.rms_count)
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

        # Ponderado los scores para buzón
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
            elif self.score_buzon < 3.0 and  (self.ah_f0 >= 8 or self.ah_rms >= 10):
                self.decision = None
            else:
                self.decision = "buzon"
        elif (self.rms_count >=70 and vad_ratio is not None and vad_ratio < 0.55 and self.score_buzon < 0 and (self.score_human - self.score_buzon) >= 2.0):
            self.decision = "humano"

    
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
                return {"decision": "buzon", "reason": f"se detecto la palabra clave '{keyword}' en la transcripcion",
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