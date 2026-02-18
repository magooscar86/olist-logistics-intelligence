"""
Configuración Global - Olist Logistics Intelligence
VERSIÓN: 3.1 (Computers Accessories añadido a Fast Rotation)
"""
from pathlib import Path

# ============================================================================
# 1. RUTAS DEL PROYECTO
# ============================================================================
import os

# Ruta dinámica - funciona en local y Google Colab
PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
MODELS_DIR = OUTPUTS_DIR / 'models'
MAPS_DIR = OUTPUTS_DIR / 'maps'
REPORTS_DIR = OUTPUTS_DIR / 'reports'

ZIP_FILE = RAW_DATA_DIR / 'archive.zip'

# ============================================================================
# 2. PARÁMETROS DE DATOS (FASE 1)
# ============================================================================
BRAZIL_BOUNDS = {
    'lat_min': -34,
    'lat_max': 6,
    'lng_min': -74,
    'lng_max': -34
}

MAX_GEO_MISSING_PCT = 1.0
MIN_WEEKS_FOR_FORECASTING = 60

# ============================================================================
# 3. FORECASTING (FASE 2)
# ============================================================================

TIME_SERIES_CONFIG = {
    'start_date': '2017-01-01',
    'end_date': '2018-08-31',
    'frequency': 'W',
    'min_weeks_required': 10
}

# KERNEL STANDARD (Categorías estables)
GP_KERNEL_CONFIG = {
    'trend_variance': 50.0**2,
    'trend_length_scale': 45.0,
    'trend_bounds': (15.0, 150.0),
    'seasonal_variance': 10.0**2,
    'seasonal_length_scale': 1.0,
    'seasonal_periodicity': 52.0,
    'periodicity_bounds': (48, 56),
    'noise_level': 10.0**2,
    'noise_bounds': (1e-2, 1e5)
}

# KERNEL FAST (Categorías rápidas)
GP_KERNEL_CONFIG_FAST = {
    'trend_variance': 50.0**2,
    'trend_length_scale': 10.0,
    'trend_bounds': (4.0, 30.0),
    'seasonal_variance': 10.0**2,
    'seasonal_length_scale': 1.0,
    'seasonal_periodicity': 52.0,
    'periodicity_bounds': (48, 56),
    'noise_level': 10.0**2,
    'noise_bounds': (1e-2, 1e5)
}

# ACTUALIZACIÓN: Añadido computers_accessories
FAST_ROTATION_CATEGORIES = [
    'fashion_bags_accessories',
    'fashion_shoes',
    'sports_leisure',
    'cool_stuff',
    'electronics',
    'computers_accessories'  # ← NUEVO: Fix warning de convergencia
]

GP_MODEL_PARAMS = {
    'normalize_y': True,
    'n_restarts_optimizer': 10,
    'random_state': 42,
    'alpha': 1e-5
}

GLOBAL_OPTIMIZER_CONFIG = {
    'maxiter': 1000,
    'popsize': 20,
    'mutation': (0.5, 1.0),
    'recombination': 0.7,
    'tol': 1e-7,
    'polish': True,
    'disp': False
}

LGBM_BASE_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'seed': 42,
    'deterministic': True,
    'force_col_wise': True,
    'feature_pre_filter': False
}

OPTUNA_CONFIG = {
    'n_trials': 40,
    'timeout': None,
    'sampler_seed': 42,
    'show_progress_bar': False
}

VALIDATION_CONFIG = {
    'test_weeks': 12,
    'confidence_level': 0.95
}

STOCK_CALCULATOR_CONFIG = {
    'z_scores': {
        'low': 1.28,
        'standard': 1.96,
        'high': 2.57
    }
}

# ============================================================================
# 4. PARÁMETROS ESPACIALES (FASE 3, 4, 5)
# ============================================================================
KRIGING_PARAMS = {
    'variogram_model': 'spherical',
    'nlags': 12,
    'weight': True
}

KMEANS_PARAMS = {
    'n_clusters': 4,
    'random_state': 42,
    'max_iter': 300
}

NETWORK_PARAMS = {
    'min_transactions': 5,
    'top_n_nodes': 20
}

# ============================================================================
# 5. VISUALIZACIÓN
# ============================================================================
COLORMAP = {
    'low_risk': '#2ecc71',
    'medium': '#f39c12',
    'high_risk': '#e74c3c'
}

MAP_CONFIG = {
    'tiles': 'OpenStreetMap',
    'zoom_start': 5,
    'location': [-14.235, -51.925]
}

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
