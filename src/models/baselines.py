"""
Módulo de Modelos Base (Baselines) - Olist Logistics Intelligence

Responsabilidad:
    - Implementar Media Móvil con ventana optimizable
    - Implementar Holt-Winters con tendencia lineal
    - Proveer interfaz estándar: fit(), predict()
    - Servir como punto de referencia para modelos complejos

Fase del Proyecto: 2 (Forecasting - Benchmark)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import optuna
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovingAverageModel:
    """Media Móvil Simple con ventana optimizable"""
    
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.history = []
        
    def optimize(self, y_train: pd.Series, y_val: pd.Series, n_trials: int = 20):
        def objective(trial):
            w = trial.suggest_int('window', 2, 12)
            hist = list(y_train.values)
            preds = []
            for val in y_val.values:
                pred = np.mean(hist[-w:])
                preds.append(pred)
                hist.append(val)
            return np.sqrt(mean_squared_error(y_val.values, preds))
            
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.window_size = study.best_params['window']
        return study.best_value

    def fit(self, y_train: pd.Series):
        self.history = list(y_train.values)
        
    def predict(self, horizon: int) -> np.ndarray:
        if not self.history:
            raise ValueError("Modelo no entrenado")
        last_ma = np.mean(self.history[-self.window_size:])
        return np.full(horizon, last_ma)
        
    def predict_walk_forward(self, y_test: pd.Series) -> np.ndarray:
        preds = []
        hist = self.history.copy()
        for val in y_test.values:
            pred = np.mean(hist[-self.window_size:])
            preds.append(pred)
            hist.append(val)
        return np.array(preds)


class HoltWintersModel:
    """Suavizamiento Exponencial (Holt Linear)"""
    
    def __init__(self, damped: bool = True):
        self.damped = damped
        self.model = None
        self.fit_model = None
        
    def optimize(self, y_train: pd.Series, y_val: pd.Series, n_trials: int = 10):
        def objective(trial):
            d = trial.suggest_categorical('damped', [True, False])
            try:
                m = ExponentialSmoothing(
                    y_train.values, 
                    trend='add', 
                    seasonal=None, 
                    damped_trend=d
                ).fit(optimized=True)
                preds = m.forecast(len(y_val))
                
                # FIX: Manejar Series y Array
                if hasattr(preds, 'values'):
                    preds = preds.values
                
                return np.sqrt(mean_squared_error(y_val.values, preds))
            except:
                return float('inf')
                
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.damped = study.best_params['damped']
        return study.best_value

    def fit(self, y_train: pd.Series):
        try:
            self.model = ExponentialSmoothing(
                y_train.values,
                trend='add',
                seasonal=None,
                damped_trend=self.damped
            )
            self.fit_model = self.model.fit(optimized=True)
        except Exception as e:
            logger.error(f"Error entrenando Holt: {e}")
            self.fit_model = None
            
    def predict(self, horizon: int) -> np.ndarray:
        """
        Predicción con tendencia (FIX aplicado)
        """
        if self.fit_model is None:
            logger.warning("Holt falló, usando fallback")
            return np.zeros(horizon)
        
        try:
            preds = self.fit_model.forecast(horizon)
            
            # FIX: Manejar tanto Series como Array
            if hasattr(preds, 'values'):
                return preds.values
            return np.array(preds)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return np.zeros(horizon)


if __name__ == '__main__':
    print("Módulo baselines.py cargado correctamente")
