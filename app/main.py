"""
AMD Service - FastAPI Server
Recibe audio via WebSocket y devuelve HUMAN/MACHINE
"""
import asyncio
import logging
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


@app.on_event("startup")
async def startup_event():
    """Cargar modelo Vosk al iniciar"""
    global amd_detector
    logger.info("Iniciando AMD Service...")
    amd_detector = AMDDetector()
    logger.info("AMD Service listo")


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
        "active_sessions": len(active_sessions)
    }


class AnalyzeRequest(BaseModel):
    """Request para analisis de audio via HTTP"""
    call_id: str
    audio_base64: str  # Audio en base64 (PCM 16-bit, 8000Hz)
    sample_rate: int = 8000


@app.post("/analyze")
async def analyze_audio(request: AnalyzeRequest):
    """
    Analiza audio completo via HTTP POST
    Util para pruebas o integracion simple

    Flujo:
    1. Detectar beep (rapido, ~10-50ms) -> Si hay beep = MACHINE
    2. Si no hay beep, transcribir con Vosk -> Analizar texto
    """
    if not amd_detector:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        # Decodificar audio
        audio_data = base64.b64decode(request.audio_base64)

        logger.info(f"[{request.call_id}] Recibido audio: {len(audio_data)} bytes")

        # ========================================
        # PASO 1: Detectar beep (muy rapido)
        # Si hay pitido = buzon de voz
        # ========================================
        beep_result = amd_detector.detect_beep(audio_data, request.sample_rate)

        if beep_result.get("detected", False):
            logger.info(f"[{request.call_id}] BEEP detectado: {beep_result}")
            result = {
                "call_id": request.call_id,
                "result": "MACHINE",
                "confidence": beep_result.get("confidence", 0.80),
                "reason": f"Pitido de buzon detectado a {beep_result.get('frequency', 0):.0f}Hz",
                "transcription": "",
                "beep_detected": True
            }
            return JSONResponse(content=result)

        logger.info(f"[{request.call_id}] No se detecto beep, continuando con transcripcion...")

        # ========================================
        # PASO 2: Si no hay beep, analizar con Vosk
        # ========================================
        # Crear sesion temporal
        session = AMDSession(amd_detector, request.call_id, request.sample_rate)

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
                logger.info(f"[{request.call_id}] Decision tomada en chunk {i // CHUNK_SIZE + 1}")
                break

        # Si no hay decision después de procesar todo, forzar
        if not result:
            logger.info(f"[{request.call_id}] Forzando decision después de procesar {len(audio_data)} bytes")
            result = session.force_decision()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error analizando audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
