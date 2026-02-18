# ============================================================================
# DOCKERFILE - Olist Logistics API
# ============================================================================

FROM python:3.10-slim

LABEL maintainer="olist-project"
LABEL description="Olist Logistics API - Sistema de predicción ML"
LABEL version="1.0.0"

WORKDIR /app

# ============================================================================
# INSTALAR DEPENDENCIAS DEL SISTEMA (incluyendo GDAL para geopandas)
# ============================================================================
RUN apt-get update && apt-get install -y \
    curl \
    gdal-bin \
    libgdal-dev \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# ============================================================================
# INSTALAR DEPENDENCIAS PYTHON
# ============================================================================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# COPIAR CÓDIGO
# ============================================================================
COPY . .

# ============================================================================
# EXPONER PUERTO
# ============================================================================
EXPOSE 8000

# ============================================================================
# COMANDO DE INICIO
# ============================================================================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]