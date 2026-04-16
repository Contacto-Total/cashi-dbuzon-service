"""
AMD Service - FastAPI Server
Recibe audio via HTTP POST o WebSocket y devuelve HUMAN/MACHINE
Mismo contrato de API que la version anterior (Vosk).
"""
import asyncio
import logging
import threading
import struct
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import numpy as np

from app.amd_detector import AMDDetector, AMDSession
from app.config import (
    SERVER_HOST,
    SERVER_PORT,
    AMD_DECISION_TIMEOUT_SECONDS,
    AUDIO_INPUT_SAMPLE_RATE,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cashi AMD Service",
    description="Deteccion de Buzon de Voz — Silero VAD + Resemblyzer + SVM",
    version="2.0.0"
)

# ── Globals ───────────────────────────────────────────────────────────────────
amd_detector: Optional[AMDDetector] = None
active_sessions: dict[str, AMDSession] = {}

# ThreadPoolExecutor: 1 thread por nucleo disponible, min 4
# Cada analisis AMD consume ~15-20MB de RAM y muy poco CPU
amd_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="amd_worker")

_analyses_in_progress = 0
_analyses_lock = threading.Lock()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global amd_detector
    logger.info("Iniciando Cashi AMD Service v2.0 (Silero VAD + Resemblyzer)...")
    amd_detector = AMDDetector()
    logger.info(f"Servicio listo. ThreadPool: {amd_executor._max_workers} workers")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Deteniendo AMD Service...")
    amd_executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor cerrado.")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "Cashi AMD Service", "version": "2.0.0"}


@app.get("/health")
async def health():
    model_ready = (
        amd_detector is not None and
        amd_detector.classifier.is_ready()
    )
    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": amd_detector is not None,
        "svm_ready": amd_detector.classifier.is_ready() if amd_detector else False,
        "active_sessions": len(active_sessions),
        "analyses_in_progress": _analyses_in_progress,
        "thread_pool_max_workers": amd_executor._max_workers,
    }


# ── HTTP POST /analyze ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    call_id: str
    audio_base64: str       # Audio PCM 16-bit en base64
    sample_rate: int = AUDIO_INPUT_SAMPLE_RATE


def _parse_wav_header(audio_data: bytes) -> dict | None:
    """Parsea WAV header, retorna info o None si no es WAV."""
    if len(audio_data) < 44:
        return None
    if audio_data[0:4] != b"RIFF" or audio_data[8:12] != b"WAVE":
        return None
    channels = struct.unpack_from("<H", audio_data, 22)[0]
    wav_sr = struct.unpack_from("<I", audio_data, 24)[0]
    bits = struct.unpack_from("<H", audio_data, 34)[0]
    return {"channels": channels, "sample_rate": wav_sr, "bits_per_sample": bits, "data_offset": 44}


def _stereo_to_mono(audio_data: bytes) -> bytes:
    """Extrae canal izquierdo (cliente) de audio estereo 16-bit PCM."""
    mono = bytearray()
    for i in range(0, len(audio_data) - 3, 4):
        mono.append(audio_data[i])
        mono.append(audio_data[i + 1])
    return bytes(mono)


def _prepare_audio(audio_data: bytes, sample_rate: int) -> tuple[bytes, int]:
    """
    Prepara el audio para el detector:
    - Salta WAV header si existe
    - Convierte estereo a mono si es necesario
    Retorna (pcm_bytes, sample_rate_efectivo)
    """
    wav_info = _parse_wav_header(audio_data)
    if wav_info:
        logger.debug(
            f"WAV header: channels={wav_info['channels']}, "
            f"sr={wav_info['sample_rate']}, bits={wav_info['bits_per_sample']}"
        )
        audio_data = audio_data[wav_info["data_offset"]:]
        sample_rate = wav_info["sample_rate"]
        if wav_info["channels"] == 2:
            audio_data = _stereo_to_mono(audio_data)
    return audio_data, sample_rate


def _process_audio_sync(call_id: str, audio_data: bytes, sample_rate: int) -> dict:
    """
    Procesamiento sincrono AMD. Se ejecuta en ThreadPoolExecutor.
    Crea una sesion temporal, procesa todo el audio en chunks y retorna decision.
    """
    audio_data, sample_rate = _prepare_audio(audio_data, sample_rate)
    logger.info(f"[{call_id}] Procesando {len(audio_data)} bytes a {sample_rate}Hz")

    session = AMDSession(amd_detector, call_id, sample_rate=sample_rate)

    # Procesar en chunks de ~250ms para simular streaming
    # 8000Hz * 2 bytes * 0.25s = 4000 bytes
    chunk_size = int(sample_rate * 2 * 0.25)
    result = None

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i: i + chunk_size]
        result = session.process_audio(chunk)
        if result:
            logger.info(f"[{call_id}] Decision en chunk {i // chunk_size + 1}")
            break

    if not result:
        result = session.force_decision()

    return result


@app.post("/analyze")
async def analyze_audio(request: AnalyzeRequest):
    """
    Analiza audio completo via HTTP POST.
    Compatible con el contrato de API anterior.
    """
    global _analyses_in_progress

    if not amd_detector:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")

    with _analyses_lock:
        _analyses_in_progress += 1
        current = _analyses_in_progress

    logger.info(f"[{request.call_id}] Solicitud recibida ({current} en proceso)")

    try:
        audio_data = base64.b64decode(request.audio_base64)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            amd_executor,
            _process_audio_sync,
            request.call_id,
            audio_data,
            request.sample_rate,
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"[{request.call_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        with _analyses_lock:
            _analyses_in_progress -= 1


# ── WebSocket /ws/{call_id} ───────────────────────────────────────────────────

@app.websocket("/ws/{call_id}")
async def websocket_amd(websocket: WebSocket, call_id: str):
    """
    WebSocket para streaming de audio en tiempo real.

    Protocolo:
      1. Cliente conecta a /ws/{call_id}
      2. Cliente envia chunks de audio binario (PCM 16-bit)
      3. Servidor responde JSON cuando toma decision y cierra conexion
    """
    await websocket.accept()
    logger.info(f"[{call_id}] WebSocket conectado")

    if not amd_detector:
        await websocket.send_json({"error": "Servicio no inicializado"})
        await websocket.close()
        return

    session = AMDSession(amd_detector, call_id, sample_rate=AUDIO_INPUT_SAMPLE_RATE)
    active_sessions[call_id] = session

    try:
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > AMD_DECISION_TIMEOUT_SECONDS:
                logger.info(f"[{call_id}] Timeout WebSocket ({elapsed:.1f}s)")
                result = session.force_decision()
                await websocket.send_json(result)
                break

            try:
                audio_data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=0.3
                )

                # Preparar audio (WAV header, estereo→mono)
                audio_data, _ = _prepare_audio(audio_data, AUDIO_INPUT_SAMPLE_RATE)

                # Procesar en executor para no bloquear el event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    amd_executor,
                    session.process_audio,
                    audio_data
                )

                if result:
                    await websocket.send_json(result)
                    break

            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        logger.info(f"[{call_id}] WebSocket desconectado por cliente")
    except Exception as e:
        logger.error(f"[{call_id}] Error WebSocket: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        active_sessions.pop(call_id, None)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[{call_id}] Sesion WebSocket cerrada")


# ── WebSocket /ws/stream ──────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket alternativo donde el call_id llega en el primer mensaje JSON.

    Protocolo:
      1. Cliente conecta a /ws/stream
      2. Cliente envia JSON: {"call_id": "xxx", "sample_rate": 8000}
      3. Servidor responde: {"status": "ready", "call_id": "xxx"}
      4. Cliente envia chunks binarios de audio
      5. Servidor responde JSON con decision y cierra
    """
    await websocket.accept()
    logger.info("WebSocket /stream conectado, esperando configuracion...")

    if not amd_detector:
        await websocket.send_json({"error": "Servicio no inicializado"})
        await websocket.close()
        return

    call_id = None
    session = None

    try:
        config = await websocket.receive_json()
        call_id = config.get("call_id", "unknown")
        sample_rate = int(config.get("sample_rate", AUDIO_INPUT_SAMPLE_RATE))

        logger.info(f"[{call_id}] Config recibida: sample_rate={sample_rate}")

        session = AMDSession(amd_detector, call_id, sample_rate=sample_rate)
        active_sessions[call_id] = session

        await websocket.send_json({"status": "ready", "call_id": call_id})

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
                    timeout=0.3
                )

                audio_data, _ = _prepare_audio(audio_data, sample_rate)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    amd_executor,
                    session.process_audio,
                    audio_data
                )

                if result:
                    await websocket.send_json(result)
                    break

            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        logger.info(f"[{call_id}] /stream desconectado")
    except Exception as e:
        logger.error(f"[{call_id}] Error /stream: {e}")
    finally:
        if call_id:
            active_sessions.pop(call_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info",
        workers=1,  # 1 worker: el modelo se carga 1 sola vez en RAM
    )