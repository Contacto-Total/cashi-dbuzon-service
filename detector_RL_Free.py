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

from amd_lr import StreamingAMD, AMDConfig, Outcome, Decision

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
LIMIT_BUFFER_MS = 2000

# 16000 bytes = 1 segundo = 1000 ms
# 1500 ms * 8000 bytes * 2 (int16) / 1000 ms = 24000 bytes
LIMIT_BUFFER_BYTES = int(
    SAMPLE_RATE_DEFAULT * 2 * (LIMIT_BUFFER_MS / 1000)
)



@app.websocket("/ws/amd-cascade/{call_id}")
async def amd_cascada_websocket(websocket: WebSocket, call_id: str):
    
    await websocket.accept()

    print(f"\nâ–¶ llamada {call_id}")

    
    # mod_audio_stream manda audio BINARIO directo (sin JSON). El disparador manda JSON primero.
    # Toleramos ambos: 1er mensaje texto -> metadata; 1er mensaje bytes -> ya es audio.
    meta = None
    primer_chunk = None
    first = await websocket.receive()
    if first.get("text"):
        meta = json.loads(first["text"])
        print(f"  metadata JSON recibida")
    elif first.get("bytes") is not None:
        primer_chunk = first["bytes"]
        print(f"  sin metadata (mod_audio_stream) — 1er frame = {len(primer_chunk)} bytes")

    cascada = CascadaAMDClass()

    amd = StreamingAMD()

    started_at = time.time()
    contador_chunk = 0

    decision = None

    # mod_audio_stream: el 1er mensaje YA puede ser audio -> procesarlo primero
    if primer_chunk is not None:
        cascada.ring_buffer.extend(primer_chunk)
        samples = np.frombuffer(primer_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        contador_chunk += 1
        decision = amd.feed_chunk(samples)

    while decision is None or decision.outcome == Outcome.UNDECIDED:
        try:
            chunk = await websocket.receive_bytes()
        except WebSocketDisconnect:
            decision = amd.force_decision()
            break

        # 1. Acumulamos el audio para el STT (PARA GPT)
        cascada.ring_buffer.extend(chunk)

        # 2. Recalculamos cada frame
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        contador_chunk += 1
        decision = amd.feed_chunk(samples)

        if decision.outcome != Outcome.UNDECIDED:   # ya decidió (BUZON/HUMANO/SEND_TO_STT)
            break
        
    # Si RL ya decidio
    elapsed_md = round((time.time() - started_at) * 1000, 2)
    event_base = {
        "ts": datetime.now().isoformat(),
        "chunk": contador_chunk,
        "elapsed": elapsed_md,
        "buffer_bytes": len(cascada.ring_buffer),
    }

    if decision.outcome == Outcome.BUZON:
        # SI es buzom
        result = {
            "decision": "buzon",
            "reason": f"RL p_human={decision.p_human:3f}",
            "transcripcion": "",}
        fuente = "RL"
    elif decision.outcome == Outcome.HUMANO:
        result = {
            "decision": "humano",
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