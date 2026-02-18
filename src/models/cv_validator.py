"""
Cross-Validation Temporal - Olist Logistics Intelligence

Responsabilidad:
    - Implementar Time Series Cross-Validation
    - Validar robustez de modelos
    - Calcular intervalos de confianza
    - Detectar overfitting

Estándar de Industria: ✅
    Amazon, Walmart, Target usan TSCV obligatoriamente
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """
    Validación cruzada temporal (expanding window)
    
    Concepto:
        - Nunca usa datos futuros para entrenar
        - Expande la ventana de entrenamiento
        - Evalúa en múltiples periodos
        
    Example:
        >>> cv = TimeSeriesCrossValidator(n_splits=5, test_size=12)
        >>> scores = cv.cross_val_score(model, X, y)
        >>> print(f"RMSE promedio: {scores.mean():.2f} ± {scores.std():.2f}")
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 12, min_train_size: int = 40):
        """
        Args:
            n_splits: Número de folds
            test_size: Tamaño del conjunto de test (semanas)
            min_train_size: Mínimo de datos para entrenar
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        
        logger.info(f"TimeSeriesCV: {n_splits} splits, test_size={test_size}")
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera índices de train/test para cada fold
        
        Args:
            X: Array de features (para obtener longitud)
            
        Returns:
            Lista de tuplas (train_indices, test_indices)
            
        Example:
            >>> for train_idx, test_idx in cv.split(X):
            >>>     X_train, X_test = X[train_idx], X[test_idx]
        """
        n_samples = len(X)
        
        # Calcular tamaño de cada step
        # Queremos que el último fold termine en n_samples
        total_range = n_samples - self.min_train_size - self.test_size
        step = total_range // (self.n_splits - 1) if self.n_splits > 1 else 0
        
        splits = []
        
        for i in range(self.n_splits):
            # Train: Desde el inicio hasta test_start
            test_start = self.min_train_size + (i * step)
            test_end = test_start + self.test_size
            
            # Validar que no nos pasemos
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, test_start)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
            
            logger.info(f"   Fold {i+1}: Train[0:{test_start}] Test[{test_start}:{test_end}]")
        
        return splits
    
    def cross_val_score(self, 
                       model,
                       X: np.ndarray,
                       y: np.ndarray,
                       metric: str = 'rmse') -> np.ndarray:
        """
        Evalúa un modelo con CV temporal
        
        Args:
            model: Modelo con métodos .fit() y .predict()
            X: Features
            y: Target
            metric: 'rmse', 'mae', o 'mape'
            
        Returns:
            Array con score de cada fold
            
        Example:
            >>> scores = cv.cross_val_score(model, X, y)
            >>> print(f"RMSE: {scores.mean():.2f} ± {scores.std():.2f}")
        """
        scores = []
        
        for i, (train_idx, test_idx) in enumerate(self.split(X), 1):
            logger.info(f"\n   Evaluando Fold {i}/{self.n_splits}...")
            
            # Split
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train
            try:
                # Detectar tipo de modelo
                if hasattr(model, 'fit'):
                    if hasattr(y_train, 'values'):
                        model.fit(y_train)  # Para MA, Holt
                    else:
                        model.fit(X_train, y_train)  # Para GP, LGBM
                
                # Predict
                if hasattr(model, 'predict_walk_forward'):
                    preds = model.predict_walk_forward(y_test)
                else:
                    if hasattr(model, 'predict') and len(X_test.shape) == 1:
                        preds = model.predict(len(y_test))
                    else:
                        preds = model.predict(X_test)
                
                # Calcular métrica
                if metric == 'rmse':
                    score = np.sqrt(mean_squared_error(y_test, preds))
                elif metric == 'mae':
                    score = mean_absolute_error(y_test, preds)
                elif metric == 'mape':
                    score = np.mean(np.abs((y_test - preds) / y_test)) * 100
                else:
                    raise ValueError(f"Métrica {metric} no soportada")
                
                scores.append(score)
                logger.info(f"      {metric.upper()}: {score:.2f}")
                
            except Exception as e:
                logger.error(f"      Error en fold {i}: {e}")
                scores.append(np.nan)
        
        return np.array(scores)
    
    def cross_val_compare(self,
                         models: Dict,
                         X: np.ndarray,
                         y: np.ndarray) -> pd.DataFrame:
        """
        Compara múltiples modelos con CV
        
        Args:
            models: Dict {nombre: modelo_instancia}
            X, y: Datos
            
        Returns:
            DataFrame con resultados
            
        Example:
            >>> models = {
            >>>     'MA': MovingAverageModel(),
            >>>     'GP': GaussianProcessForecaster()
            >>> }
            >>> results = cv.cross_val_compare(models, X, y)
        """
        results = []
        
        for name, model in models.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluando: {name}")
            logger.info(f"{'='*70}")
            
            scores = self.cross_val_score(model, X, y)
            
            results.append({
                'Modelo': name,
                'RMSE_mean': scores.mean(),
                'RMSE_std': scores.std(),
                'RMSE_min': scores.min(),
                'RMSE_max': scores.max(),
                'Folds': len(scores[~np.isnan(scores)])
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('RMSE_mean')
        
        return df_results


if __name__ == '__main__':
    print("Módulo cv_validator.py cargado")
