"""
Módulo de Modelo de Proceso Gaussiano (DEBUG MODE + KEY FIX)
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C
import logging
from typing import Tuple, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianProcessForecaster:
    
    def __init__(self, use_fast_config: bool = False, custom_kernel_config: Optional[Dict] = None):
        if custom_kernel_config is not None:
            self.kernel_cfg = custom_kernel_config
        elif use_fast_config:
            from config import GP_KERNEL_CONFIG_FAST
            self.kernel_cfg = GP_KERNEL_CONFIG_FAST
        else:
            from config import GP_KERNEL_CONFIG
            self.kernel_cfg = GP_KERNEL_CONFIG
        
        from config import GP_MODEL_PARAMS
        self.model_params = GP_MODEL_PARAMS
        self.model = None
        self.is_trained = False
        
    def _build_kernel(self) -> C:
        cfg = self.kernel_cfg
        
        # 1. Tendencia (RBF)
        k_trend = C(cfg['trend_variance']) * RBF(
            length_scale=cfg['trend_length_scale'],
            length_scale_bounds=cfg['trend_bounds']
        )
        
        # 2. Estacionalidad
        # CORRECCIÓN AQUÍ: Usamos 'seasonal_periodicity' que es como está en config.py
        k_seasonal = C(cfg['seasonal_variance']) * ExpSineSquared(
            length_scale=cfg['seasonal_length_scale'],
            periodicity=cfg['seasonal_periodicity'], 
            periodicity_bounds=cfg['periodicity_bounds']
        )
        
        # 3. Ruido
        k_noise = WhiteKernel(
            noise_level=cfg['noise_level'],
            noise_level_bounds=cfg['noise_bounds']
        )
        
        return k_trend + k_seasonal + k_noise

    def train(self, X: np.ndarray, y: np.ndarray):
        logger.info("🧠 Entrenando GP (Debug Mode)...")
        kernel = self._build_kernel()
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42,
            alpha=1e-5
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # --- DEBUGGING ---
        logger.info(f"   Kernel Final: {self.model.kernel_}")
        
        # Insights
        insights = self.get_insights()
        memoria = insights.get('memoria_predictiva', 0)
        logger.info(f"   📊 Memoria detectada: {memoria:.2f}")

    def predict(self, X: np.ndarray, return_std: bool = True):
        if not self.is_trained: raise RuntimeError("No entrenado")
        return self.model.predict(X, return_std=return_std)

    def get_insights(self) -> Dict[str, float]:
        if not self.is_trained: return {}
        params = self.model.kernel_.get_params()
        insights = {}
        
        for key, val in params.items():
            if 'length_scale' in key and 'periodicity' not in key:
                if isinstance(val, (int, float, np.number)):
                    insights['candidate_ls'] = float(val)
                    # Verificar si está dentro de los bounds de tendencia
                    bounds = self.kernel_cfg['trend_bounds']
                    if bounds[0] <= val <= bounds[1]:
                         insights['memoria_predictiva'] = float(val)
        
        # Fallback
        if 'memoria_predictiva' not in insights and 'candidate_ls' in insights:
             insights['memoria_predictiva'] = insights['candidate_ls']
                
        return insights
