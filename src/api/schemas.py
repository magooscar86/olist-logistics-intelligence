"""
Schemas de validación para la API - Olist Logistics

Pydantic valida automáticamente:
- Tipos de datos
- Rangos permitidos
- Campos obligatorios
- Formatos

Si algo está mal, FastAPI retorna error 422 automáticamente.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal


# ============================================================================
# SCHEMAS PARA PREDICCIÓN ESPACIAL
# ============================================================================

class ShippingRequest(BaseModel):
    """
    Request para predicción de envío
    
    Example:
        {
            "latitude": -23.55,
            "longitude": -46.63
        }
    """
    latitude: float = Field(
        ...,  # ... significa "obligatorio"
        ge=-90,  # greater or equal (mayor o igual)
        le=90,   # less or equal (menor o igual)
        description="Latitud en grados decimales",
        example=-23.55
    )
    
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitud en grados decimales",
        example=-46.63
    )
    
    # Validador custom para Brasil
    @validator('latitude')
    def validate_brazil_latitude(cls, v):
        """Brasil está entre 5°N y 34°S"""
        if not (-34 <= v <= 5):
            raise ValueError('Coordinates outside Brazil')
        return v
    
    @validator('longitude')
    def validate_brazil_longitude(cls, v):
        """Brasil está entre 35°W y 74°W"""
        if not (-74 <= v <= -35):
            raise ValueError('Coordinates outside Brazil')
        return v


class ShippingResponse(BaseModel):
    """
    Response de predicción de envío
    
    Example:
        {
            "delay_days": -5.2,
            "interpretation": "ANTICIPADA",
            "confidence_level": "HIGH",
            "uncertainty_sigma": 2.1,
            "nearest_city": "Sao Paulo",
            "recommendation": "Safe to promise fast delivery"
        }
    """
    delay_days: float = Field(
        ...,
        description="Días de retraso predichos (negativo = anticipación)"
    )
    
    interpretation: Literal["ANTICIPADA", "RETRASO"] = Field(
        ...,
        description="Interpretación del resultado"
    )
    
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Nivel de confianza de la predicción"
    )
    
    uncertainty_sigma: float = Field(
        ...,
        description="Incertidumbre del modelo (sigma)"
    )
    
    nearest_city: str = Field(
        ...,
        description="Ciudad más cercana a las coordenadas"
    )
    
    recommendation: str = Field(
        ...,
        description="Recomendación operativa"
    )


# ============================================================================
# SCHEMAS PARA PREDICCIÓN DE DEMANDA (para después)
# ============================================================================

class DemandRequest(BaseModel):
    """Request para predicción de demanda"""
    category: str = Field(
        ...,
        description="Categoría de producto",
        example="health_beauty"
    )
    
    horizon: int = Field(
        ...,
        ge=1,
        le=52,
        description="Horizonte de predicción en semanas",
        example=12
    )
    
    service_level: Optional[float] = Field(
        0.95,
        ge=0.5,
        le=0.99,
        description="Nivel de servicio para stock de seguridad",
        example=0.95
    )


class DemandResponse(BaseModel):
    """Response de predicción de demanda"""
    category: str
    horizon: int
    prediction: float = Field(..., description="Demanda predicha")
    lower_bound: float = Field(..., description="Límite inferior (IC)")
    upper_bound: float = Field(..., description="Límite superior (IC)")
    safety_stock: float = Field(..., description="Stock de seguridad")
    model_used: str = Field(..., description="Modelo utilizado")


# ============================================================================
# SCHEMA GENÉRICO DE ERROR
# ============================================================================

class ErrorResponse(BaseModel):
    """Response de error estándar"""
    error: str = Field(..., description="Tipo de error")
    message: str = Field(..., description="Mensaje descriptivo")
    details: Optional[dict] = Field(None, description="Detalles adicionales")
