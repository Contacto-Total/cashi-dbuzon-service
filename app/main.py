"""
AMD Service - FastAPI Server
Recibe audio via WebSocket y devuelve HUMAN/MACHINE
"""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64

from app.amd_detector import AMDDetector, AMDSession
from app.config import SERVER_HOST, SERVER_PORT, AMD_DECISION_TIMEOUT_SECONDS


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="AMD Service",
    description="Servicio de Deteccion de Maquina Contestadora (Buzon de Voz)",
    version="1.0.0"
)

# Inicializar detector AMD (singleton)
amd_detector: Optional[AMDDetector] = None

# Sesiones activas
active_sessions: dict[str, AMDSession] = {}

# ThreadPoolExecutor para procesamiento paralelo de AMD
# 3 threads por worker = 6 analisis paralelos con 2 workers
amd_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="amd_worker")

# Contador thread-safe para load shedding
_analyses_in_progress = 0
_analyses_lock = threading.Lock()


@app.on_event("startup")
async def startup_event():
    """Cargar modelo Vosk al iniciar"""
    global amd_detector
    logger.info("Iniciando AMD Service...")
    logger.info(f"ThreadPoolExecutor configurado con {amd_executor._max_workers} workers")
    amd_detector = AMDDetector()
    logger.info("AMD Service listo")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al detener"""
    logger.info("Deteniendo AMD Service...")
    amd_executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor cerrado")


@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "AMD Service", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check detallado"""
    return {
        "status": "healthy",
        "model_loaded": amd_detector is not None,
        "active_sessions": len(active_sessions),
        "thread_pool_max_workers": amd_executor._max_workers,
        "thread_pool_threads": len(amd_executor._threads) if hasattr(amd_executor, '_threads') else 0,
        "analyses_in_progress": _analyses_in_progress
    }


class AnalyzeRequest(BaseModel):
    """Request para analisis de audio via HTTP"""
    call_id: str
    audio_base64: str  # Audio en base64 (PCM 16-bit, 8000Hz)
    sample_rate: int = 8000


def _stereo_to_mono(audio_data: bytes) -> bytes:
    """
    Convierte audio estéreo a mono extrayendo solo el canal izquierdo (cliente).

    En grabaciones de llamadas FreeSWITCH con record_stereo:
    - Canal izquierdo (bytes 0-1): el cliente (far-end) - ESTE ES EL QUE NECESITAMOS
    - Canal derecho (bytes 2-3): nuestro lado (agente/sistema) - vacío antes del bridge

    Para AMD solo nos interesa el canal del cliente (izquierdo).

    Formato: 16-bit PCM, little-endian
    Estéreo: L0 L1 R0 R1 L0 L1 R0 R1 ... (4 bytes por frame)
    Mono:    L0 L1 L0 L1 ... (2 bytes por frame)
    """
    if len(audio_data) < 4:
        return audio_data

    # Extraer solo el canal IZQUIERDO (cada 4 bytes, tomar bytes 0-1)
    mono_data = bytearray()
    for i in range(0, len(audio_data) - 3, 4):
        # Canal izquierdo = bytes 0 y 1 de cada frame de 4 bytes
        mono_data.append(audio_data[i])
        mono_data.append(audio_data[i + 1])

    return bytes(mono_data)


def _parse_wav_header(audio_data: bytes) -> dict:
    """
    Parsea WAV header para obtener info del audio.
    Returns dict con channels, sample_rate, bits_per_sample, data_offset
    o None si no es WAV.
    """
    if len(audio_data) < 44:
        return None
    # Verificar magic bytes: RIFF....WAVE
    if audio_data[0:4] != b'RIFF' or audio_data[8:12] != b'WAVE':
        return None
    import struct
    channels = struct.unpack_from('<H', audio_data, 22)[0]
    wav_sample_rate = struct.unpack_from('<I', audio_data, 24)[0]
    bits_per_sample = struct.unpack_from('<H', audio_data, 34)[0]
    return {
        "channels": channels,
        "sample_rate": wav_sample_rate,
        "bits_per_sample": bits_per_sample,
        "data_offset": 44
    }


def _process_audio_sync(call_id: str, audio_data: bytes, sample_rate: int) -> dict:
    """
    Funcion sincrona para procesar audio AMD.
    Se ejecuta en ThreadPoolExecutor para permitir multiples analisis en paralelo.

    Flujo:
    1. Detectar WAV header y extraer PCM raw
    2. Convertir estéreo a mono SOLO si el audio es estéreo
    3. Detectar beep (rapido, ~10-50ms) -> Si hay beep = MACHINE
    4. Si no hay beep, transcribir con Vosk -> Analizar texto
    """
    original_size = len(audio_data)

    # PASO 0: Detectar y saltar WAV header si existe
    wav_info = _parse_wav_header(audio_data)
    if wav_info:
        logger.info(f"[{call_id}] WAV header detectado: channels={wav_info['channels']}, "
                     f"sample_rate={wav_info['sample_rate']}, bits={wav_info['bits_per_sample']}")
        audio_data = audio_data[wav_info['data_offset']:]  # Saltar header de 44 bytes
        num_channels = wav_info['channels']
    else:
        logger.info(f"[{call_id}] Sin WAV header, asumiendo PCM raw mono")
        num_channels = 1

    # Solo convertir estéreo a mono si REALMENTE es estéreo (2 canales)
    # RECORD_STEREO=false en predictivo = audio ya es MONO, no convertir
    if num_channels == 2 and len(audio_data) >= 8:
        audio_data = _stereo_to_mono(audio_data)
        logger.info(f"[{call_id}] Convertido estéreo a mono: {original_size} -> {len(audio_data)} bytes")
    else:
        logger.info(f"[{call_id}] Audio ya es mono, sin conversión ({len(audio_data)} bytes PCM)")

    logger.info(f"[{call_id}] Procesando audio: {len(audio_data)} bytes")

    # ========================================
    # PASO 1: Detectar beep (muy rapido)
    # Si hay pitido = buzon de voz
    # ========================================
    beep_result = amd_detector.detect_beep(audio_data, sample_rate)

    if beep_result.get("detected", False):
        logger.info(f"[{call_id}] BEEP detectado: {beep_result}")
        return {
            "call_id": call_id,
            "result": "MACHINE",
            "confidence": beep_result.get("confidence", 0.80),
            "reason": f"Pitido de buzon detectado a {beep_result.get('frequency', 0):.0f}Hz",
            "transcription": "",
            "beep_detected": True
        }

    logger.info(f"[{call_id}] No se detecto beep, continuando con transcripcion...")

    # ========================================
    # PASO 2: Si no hay beep, analizar con Vosk
    # ========================================
    # Crear sesion temporal (cada sesion tiene su propio KaldiRecognizer)
    session = AMDSession(amd_detector, call_id, sample_rate)

    # IMPORTANTE: Vosk necesita recibir audio en chunks pequeños
    # No puede procesar todo de golpe. Chunk size = 4000 bytes (~0.25s a 8kHz 16-bit)
    CHUNK_SIZE = 4000
    result = None

    # Procesar audio en chunks para que Vosk pueda analizarlo correctamente
    for i in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i:i + CHUNK_SIZE]
        result = session.process_audio(chunk)

        # Si ya tomó una decisión, salir del loop
        if result:
            logger.info(f"[{call_id}] Decision tomada en chunk {i // CHUNK_SIZE + 1}")
            break

    # Si no hay decision después de procesar todo, forzar
    if not result:
        logger.info(f"[{call_id}] Forzando decision después de procesar {len(audio_data)} bytes")
        result = session.force_decision()

    return result


@app.post("/analyze")
async def analyze_audio(request: AnalyzeRequest):
    """
    Analiza audio completo via HTTP POST
    Util para pruebas o integracion simple

    El procesamiento se ejecuta en ThreadPoolExecutor permitiendo
    multiples analisis AMD en paralelo (3 por worker).

    Las llamadas que excedan la capacidad se encolan automaticamente
    en el ThreadPoolExecutor hasta que se libere un slot.
    """
    global _analyses_in_progress

    if not amd_detector:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    # Incrementar contador (para monitoreo)
    with _analyses_lock:
        _analyses_in_progress += 1
        current_count = _analyses_in_progress

    logger.info(f"[{request.call_id}] Analisis encolado ({current_count} en proceso/cola)")

    try:
        # Decodificar audio
        audio_data = base64.b64decode(request.audio_base64)

        logger.info(f"[{request.call_id}] Recibido audio: {len(audio_data)} bytes")

        # Ejecutar procesamiento en thread pool para permitir paralelismo
        # Si todos los threads estan ocupados, la tarea se encola automaticamente
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            amd_executor,
            _process_audio_sync,
            request.call_id,
            audio_data,
            request.sample_rate
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error analizando audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Decrementar contador
        with _analyses_lock:
            _analyses_in_progress -= 1


@app.websocket("/ws/{call_id}")
async def websocket_amd(websocket: WebSocket, call_id: str):
    """
    WebSocket para streaming de audio en tiempo real

    Protocolo:
    1. Cliente conecta a /ws/{call_id}
    2. Cliente envia chunks de audio (PCM 16-bit, 8000Hz)
    3. Servidor responde con JSON cuando toma decision:
       {"result": "HUMAN"|"MACHINE"|"UNKNOWN", "confidence": 0.0-1.0, ...}
    4. Conexion se cierra despues de la decision
    """
    await websocket.accept()
    logger.info(f"[{call_id}] WebSocket conectado")

    if not amd_detector:
        await websocket.send_json({"error": "Modelo no cargado"})
        await websocket.close()
        return

    # Crear sesion AMD
    session = AMDSession(amd_detector, call_id, sample_rate=8000)
    active_sessions[call_id] = session

    try:
        # Timeout para decision
        start_time = asyncio.get_event_loop().time()

        while True:
            # Verificar timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > AMD_DECISION_TIMEOUT_SECONDS:
                logger.info(f"[{call_id}] Timeout alcanzado ({elapsed:.1f}s)")
                result = session.force_decision()
                await websocket.send_json(result)
                break

            try:
                # Recibir audio con timeout corto
                audio_data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=0.5
                )

                # Procesar audio
                result = session.process_audio(audio_data)

                if result:
                    # Decision tomada!
                    await websocket.send_json(result)
                    break

            except asyncio.TimeoutError:
                # No hay datos, continuar esperando
                continue

    except WebSocketDisconnect:
        logger.info(f"[{call_id}] WebSocket desconectado por cliente")
    except Exception as e:
        logger.error(f"[{call_id}] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Limpiar sesion
        if call_id in active_sessions:
            del active_sessions[call_id]
        try:
            await websocket.close()
        except:
            pass
        logger.info(f"[{call_id}] Sesion cerrada")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket alternativo donde el call_id viene en el primer mensaje

    Protocolo:
    1. Cliente conecta a /ws/stream
    2. Cliente envia JSON: {"call_id": "xxx", "sample_rate": 8000}
    3. Cliente envia chunks de audio binario
    4. Servidor responde con JSON cuando toma decision
    """
    await websocket.accept()
    logger.info("WebSocket stream conectado, esperando configuracion...")

    if not amd_detector:
        await websocket.send_json({"error": "Modelo no cargado"})
        await websocket.close()
        return

    call_id = None
    session = None

    try:
        # Primer mensaje: configuracion
        config = await websocket.receive_json()
        call_id = config.get("call_id", "unknown")
        sample_rate = config.get("sample_rate", 8000)

        logger.info(f"[{call_id}] Configuracion recibida: sample_rate={sample_rate}")

        # Crear sesion
        session = AMDSession(amd_detector, call_id, sample_rate)
        active_sessions[call_id] = session

        # Confirmar
        await websocket.send_json({"status": "ready", "call_id": call_id})

        # Timeout para decision
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > AMD_DECISION_TIMEOUT_SECONDS:
                result = session.force_decision()
                await websocket.send_json(result)
                break

            try:
                audio_data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=0.5
                )

                result = session.process_audio(audio_data)
                if result:
                    await websocket.send_json(result)
                    break

            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        logger.info(f"[{call_id}] WebSocket desconectado")
    except Exception as e:
        logger.error(f"[{call_id}] Error: {e}")
    finally:
        if call_id and call_id in active_sessions:
            del active_sessions[call_id]
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info"
    )
