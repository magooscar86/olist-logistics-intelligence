"""
Utilidades para la API - Olist Logistics

Funciones helper para:
- Cargar modelos del registry
- Hacer predicciones
- Procesar resultados
"""

import sys
from pathlib import Path

# Agregar path del proyecto
# Detectar si estamos en Docker o Colab

import os
if os.path.exists('/app'):
    # Estamos en Docker
    project_root = Path('/app')
else:
    # Estamos en local/Jupyter - usar ruta relativa desde este archivo
    # utils.py está en: src/api/utils.py
    # Necesitamos subir 3 niveles: api -> src -> Olist_Project
    project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.model_registry import ModelRegistry
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SINGLETON PARA MODELOS (carga una sola vez)
# ============================================================================

class ModelCache:
    """Cache de modelos para no recargar en cada request"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def load_spatial_model(self):
        """Carga modelo espacial si no está en cache"""
        if 'gp_spatial' not in self._models:
            logger.info("Cargando modelo espacial...")
            registry = ModelRegistry()
            self._models['gp_spatial'] = registry.load_latest('gp_spatial', 'spatial')
            logger.info("✅ Modelo espacial cargado")
        return self._models['gp_spatial']
    
    def load_df_main(self):
        """Carga df_main si no está en cache"""
        if 'df_main' not in self._models:
            logger.info("Cargando df_main...")
            df_path = project_root / 'checkpoints' / 'df_main.parquet'
            self._models['df_main'] = pd.read_parquet(df_path)
            logger.info("✅ df_main cargado")
        return self._models['df_main']


# Instancia global
model_cache = ModelCache()


# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def predict_shipping_delay(latitude: float, longitude: float) -> dict:
    """
    Predice delay de envío usando GP espacial
    
    Args:
        latitude: Latitud en grados decimales
        longitude: Longitud en grados decimales
        
    Returns:
        dict con predicción e interpretación
    """
    # Cargar modelo y datos
    gp_spatial = model_cache.load_spatial_model()
    df_main = model_cache.load_df_main()
    
    # Preparar input
    X_new = np.array([[latitude, longitude]])
    
    # Predecir
    y_pred, sigma = gp_spatial.predict(X_new, return_std=True)
    
    delay_days = float(y_pred[0])
    uncertainty = float(sigma[0])
    
    # Interpretación
    if delay_days < 0:
        interpretation = "ANTICIPADA"
        recommendation = f"Safe to promise fast delivery. Buffer: {abs(delay_days):.1f} days."
    else:
        interpretation = "RETRASO"
        recommendation = f"Add {int(delay_days)+1} extra days to standard ETA."
    
    # Nivel de confianza
    if uncertainty < 3.0:
        confidence = "HIGH"
    elif uncertainty < 6.0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Geocoding inverso (ciudad más cercana)
    nearest_city = get_nearest_city(latitude, longitude, df_main)
    
    return {
        "delay_days": round(delay_days, 2),
        "interpretation": interpretation,
        "confidence_level": confidence,
        "uncertainty_sigma": round(uncertainty, 2),
        "nearest_city": nearest_city,
        "recommendation": recommendation
    }


def get_nearest_city(lat: float, lng: float, df: pd.DataFrame) -> str:
    """
    Encuentra ciudad más cercana
    
    Args:
        lat: Latitud
        lng: Longitud
        df: DataFrame con coordenadas
        
    Returns:
        Nombre de la ciudad
    """
    try:
        # Calcular distancias
        distances = (df['geolocation_lat'] - lat)**2 + \
                   (df['geolocation_lng'] - lng)**2
        
        idx_min = distances.idxmin()
        city = df.loc[idx_min, 'geolocation_city']
        
        return city.title()
    except Exception as e:
        logger.error(f"Error en geocoding: {e}")
        return "Unknown"


# ============================================================================
# FUNCIÓN PARA INICIALIZAR MODELOS AL STARTUP
# ============================================================================

def warmup_models():
    """
    Carga modelos al iniciar la API
    Evita delay en el primer request
    """
    logger.info("🔥 Warmup: Precargando modelos...")
    
    try:
        model_cache.load_spatial_model()
        model_cache.load_df_main()
        logger.info("✅ Todos los modelos precargados")
        return True
    except Exception as e:
        logger.error(f"❌ Error en warmup: {e}")
        return False
