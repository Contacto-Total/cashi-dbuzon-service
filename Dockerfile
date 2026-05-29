FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Descargar modelo Vosk espanol
RUN mkdir -p models && \
    cd models && \
    wget -q https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip && \
    unzip vosk-model-small-es-0.42.zip && \
    rm vosk-model-small-es-0.42.zip

# Copiar codigo
COPY app/ ./app/

# Puerto
EXPOSE 8765

# Variables de entorno
ENV VOSK_MODEL_PATH=models/vosk-model-small-es-0.42
ENV AMD_HOST=0.0.0.0
ENV AMD_PORT=8765

# Ejecutar
CMD ["python", "-m", "app.main"]
