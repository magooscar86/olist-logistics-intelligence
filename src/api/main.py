"""
Olist Logistics API - Versión con Modelos Reales

Endpoints:
- GET  /         : Bienvenida
- GET  /health   : Health check
- POST /predict/shipping : Predicción de delay espacial (REAL)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

# Importar schemas y utils
from src.api.schemas import (
    ShippingRequest,
    ShippingResponse,
    ErrorResponse
)
from src.api.utils import predict_shipping_delay, warmup_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CREAR APP
# ============================================================================

app = FastAPI(
    title="Olist Logistics API",
    description="Sistema de predicción de demanda y envíos con ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# EVENTO DE STARTUP (precarga modelos)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la API"""
    logger.info("🚀 Iniciando Olist Logistics API...")
    
    # Precargar modelos
    success = warmup_models()
    
    if success:
        logger.info("✅ API lista para recibir requests")
    else:
        logger.warning("⚠️ API iniciada pero modelos no precargados")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Endpoint de bienvenida"""
    return {
        "message": "¡Bienvenido a Olist Logistics API!",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict_shipping": "/predict/shipping"
        }
    }


@app.get("/health")
def health_check():
    """Verifica estado del sistema"""
    from src.api.utils import model_cache
    
    models_loaded = {
        "gp_spatial": 'gp_spatial' in model_cache._models,
        "df_main": 'df_main' in model_cache._models
    }
    
    all_loaded = all(models_loaded.values())
    
    return {
        "status": "healthy" if all_loaded else "degraded",
        "service": "olist-api",
        "models_loaded": models_loaded
    }


@app.post(
    "/predict/shipping",
    response_model=ShippingResponse,
    responses={
        200: {"description": "Predicción exitosa"},
        422: {"description": "Datos de entrada inválidos"},
        500: {"description": "Error interno del servidor"}
    }
)
def predict_shipping(request: ShippingRequest):
    """
    Predice delay de envío para coordenadas específicas
    
    Usa modelo GP espacial entrenado con Kriging.
    
    Args:
        request: Objeto con latitude y longitude
        
    Returns:
        Predicción de delay con interpretación
        
    Example:
        Request:
```json
        {
            "latitude": -23.55,
            "longitude": -46.63
        }
```
        
        Response:
```json
        {
            "delay_days": -5.2,
            "interpretation": "ANTICIPADA",
            "confidence_level": "HIGH",
            "uncertainty_sigma": 2.1,
            "nearest_city": "Sao Paulo",
            "recommendation": "Safe to promise fast delivery..."
        }
```
    """
    try:
        # Hacer predicción
        result = predict_shipping_delay(
            latitude=request.latitude,
            longitude=request.longitude
        )
        
        return result
        
    except ValueError as e:
        # Error de validación (ej: fuera de Brasil)
        raise HTTPException(
            status_code=422,
            detail={"error": "ValidationError", "message": str(e)}
        )
    
    except Exception as e:
        # Error inesperado
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": "Error al procesar predicción"}
        )


# ============================================================================
# MANEJADOR DE ERRORES GLOBAL
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Captura errores no manejados"""
    logger.error(f"Error no manejado: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Ocurrió un error inesperado",
            "details": str(exc)
        }
    )
