"""
Quick Load Helper - Carga rápida de datos y modelos

Uso en notebooks:
    from notebooks.quick_load import quick_load
    
    data = quick_load()
    
    # Acceder a datos
    df_main = data['df_main']
    gp_spatial = data['gp_spatial']
    cv_results = data['cv_results']
"""

import sys
from pathlib import Path

# Setup path - auto-detect environment (relative path for portability)
try:
    from google.colab import drive
    project_root = Path('/content/drive/MyDrive/Olist_Project')
except ImportError:
    # Ruta local relativa - funciona desde cualquier ubicación
    project_root = Path.cwd()
    # Si estamos en notebooks/, subir un nivel
    if project_root.name == 'notebooks':
        project_root = project_root.parent
        
sys.path.insert(0, str(project_root))

from src.utils.model_registry import ModelRegistry
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_load(load_spatial=True, load_forecasting=False):
    """
    Carga rápida de datos y modelos desde cache/registry
    
    Args:
        load_spatial: Si cargar modelo espacial
        load_forecasting: Si cargar modelos de forecasting
        
    Returns:
        Dict con todos los datos cargados
    """
    logger.info("="*70)
    logger.info("⚡ QUICK LOAD - CARGA RÁPIDA")
    logger.info("="*70)
    
    registry = ModelRegistry()
    data = {}
    
    # Cargar df_main
    logger.info("\n[1/5] Cargando df_main...")
    cache_path = project_root / 'cache' / 'df_main.pkl'
    
    if cache_path.exists():
        import joblib
        df_path = joblib.load(cache_path)
        data['df_main'] = pd.read_parquet(df_path)
        logger.info(f"   ✅ df_main: {data['df_main'].shape}")
    else:
        logger.warning("   ⚠️ df_main no encontrado en cache")
    
    # Cargar spatial data
    logger.info("\n[2/5] Cargando datos espaciales...")
    cache_path = project_root / 'cache' / 'spatial_data.pkl'
    
    if cache_path.exists():
        import joblib
        df_path = joblib.load(cache_path)
        data['df_train_geo'] = pd.read_parquet(df_path)
        logger.info(f"   ✅ df_train_geo: {data['df_train_geo'].shape}")
    
    cache_path = project_root / 'cache' / 'df_spatial.pkl'
    if cache_path.exists():
        import joblib
        df_path = joblib.load(cache_path)
        data['df_spatial'] = pd.read_parquet(df_path)
        logger.info(f"   ✅ df_spatial: {data['df_spatial'].shape}")
    
    # Cargar CV results
    logger.info("\n[3/5] Cargando resultados de CV...")
    cv_cached = registry.load_cache('cv_results')
    if cv_cached is not None:
        data['cv_results'] = cv_cached
        logger.info(f"   ✅ cv_results: {len(cv_cached)} filas")
    
    selection_cached = registry.load_cache('model_selection')
    if selection_cached is not None:
        data['model_selection'] = selection_cached
        logger.info(f"   ✅ model_selection: {len(selection_cached)} filas")
    
    # Cargar modelo espacial
    if load_spatial:
        logger.info("\n[4/5] Cargando modelo espacial...")
        try:
            data['gp_spatial'] = registry.load_latest('gp_spatial', 'spatial')
            logger.info("   ✅ gp_spatial cargado")
            
            # Mostrar métricas
            versions = registry.list_versions('gp_spatial', 'spatial')
            if versions:
                latest = versions[-1]
                logger.info(f"   📊 Métricas: {latest['metrics']}")
        except FileNotFoundError:
            logger.warning("   ⚠️ gp_spatial no encontrado")
    
    # Cargar modelos de forecasting
    if load_forecasting:
        logger.info("\n[5/5] Cargando modelos de forecasting...")
        data['forecasting_models'] = {}
        
        # Listar todos los modelos disponibles
        metadata = registry.metadata.get('forecasting', {})
        
        for model_name in metadata.keys():
            try:
                model = registry.load_latest(model_name, 'forecasting')
                data['forecasting_models'][model_name] = model
                logger.info(f"   ✅ {model_name}")
            except:
                pass
    
    logger.info("\n" + "="*70)
    logger.info("✅ CARGA COMPLETADA")
    logger.info("="*70)
    
    logger.info(f"\n📦 Datos cargados: {list(data.keys())}")
    
    return data


if __name__ == '__main__':
    # Prueba
    data = quick_load()
    print("\nPrueba exitosa!")
