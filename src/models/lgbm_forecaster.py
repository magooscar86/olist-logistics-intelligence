"""
Módulo de Pronóstico Avanzado (LightGBM) - Olist Logistics Intelligence

Responsabilidad:
    - Feature Engineering: Convertir series temporales en datos tabulares
    - LightGBM Puro: Modelo supervisado con lags y rolling
    - LightGBM Híbrido: Combina señal GP con LightGBM
    - Optimización automática de hiperparámetros con Optuna

Fase del Proyecto: 2 (Forecasting Avanzado)

Conceptos Clave:
    - Feature Engineering: Crear variables predictivas desde la serie
    - Lags: Valores históricos (t-1, t-2, t-4)
    - Rolling: Promedios y volatilidades móviles
    - Híbrido: GP captura tendencia macro + LGBM ajusta detalles
    - Optuna: Búsqueda bayesiana de mejores hiperparámetros
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LGBMFeatureEngineer:
    """
    Generador de features para modelos supervisados
    
    Concepto:
        LightGBM necesita datos tabulares (filas x columnas).
        Convertimos series temporales en tabla con:
        - Calendario (semana, mes)
        - Historia (lags)
        - Tendencias (rolling)
        - (Opcional) Señal GP
        
    Example:
        >>> fe = LGBMFeatureEngineer()
        >>> df_features = fe.create_features(df_ts)
        >>> # Resultado:
        >>> # ventas | semana | mes | lag_1 | lag_2 | roll_mean_4
        >>> #   140  |   5    |  2  |  130  |  110  |    115
    """
    
    @staticmethod
    def create_features(df_ts: pd.DataFrame, 
                       gp_signal: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Genera features desde serie temporal
        
        Args:
            df_ts: DataFrame con columna 'ventas' e índice datetime
            gp_signal: (Opcional) Señal del GP para modelo híbrido
            
        Returns:
            DataFrame con features listos para LGBM
            
        Process:
            1. Extrae info del calendario
            2. Crea lags (t-1, t-2, t-4)
            3. Crea rolling windows
            4. (Opcional) Añade GP signal
            5. Elimina NaNs (filas incompletas)
        """
        df = df_ts.copy()
        
        logger.info("   🛠️  Creando features...")
        
        # 1. Features de Calendario
        # (LightGBM no entiende fechas, necesita números)
        df['semana'] = df.index.isocalendar().week.astype(int)
        df['mes'] = df.index.month
        
        # 2. Lags (Inercia del pasado)
        # lag_1: ¿Cuánto vendí la semana pasada?
        df['lag_1'] = df['ventas'].shift(1)
        df['lag_2'] = df['ventas'].shift(2)
        df['lag_4'] = df['ventas'].shift(4)  # Hace ~1 mes
        
        # 3. Rolling Windows (Tendencias suavizadas)
        # Promedio de las últimas 4 semanas
        df['roll_mean_4'] = df['ventas'].shift(1).rolling(4, min_periods=1).mean()
        # Volatilidad (desviación estándar)
        df['roll_std_4'] = df['ventas'].shift(1).rolling(4, min_periods=1).std()
        
        # 4. GP Signal (Solo para modelo híbrido)
        if gp_signal is not None:
            if len(gp_signal) != len(df):
                raise ValueError(
                    f"GP Signal length mismatch: {len(gp_signal)} vs {len(df)}"
                )
            df['gp_signal'] = gp_signal
            logger.info("      ✅ GP Signal añadido (modo HÍBRIDO)")
        
        # 5. Eliminar NaNs
        # (Las primeras filas tienen NaNs por los lags)
        df_clean = df.dropna()
        
        n_features = len(df_clean.columns) - 1  # -1 por 'ventas'
        logger.info(f"      ✅ {n_features} features creados | {len(df_clean)} filas válidas")
        
        return df_clean


class LGBMForecaster:
    """
    Modelo LightGBM con optimización automática
    
    Capacidades:
        - Modo PURO: Solo features temporales
        - Modo HÍBRIDO: Incluye señal GP
        - Optimización bayesiana (Optuna)
        - Early stopping automático
        
    Ventajas:
        - Muy preciso en patrones complejos
        - Rápido de entrenar
        - Maneja no-linealidades
        
    Desventajas:
        - Necesita feature engineering
        - Puede sobreajustar con pocos datos
        - No extrapola tendencias (solo interpola)
        
    Example:
        >>> # LGBM Puro
        >>> lgbm_pure = LGBMForecaster(hybrid=False)
        >>> lgbm_pure.optimize(X_train, y_train, X_val, y_val)
        >>> lgbm_pure.fit(X_train, y_train)
        >>> predictions = lgbm_pure.predict(X_test)
        >>>
        >>> # LGBM Híbrido (con GP)
        >>> lgbm_hyb = LGBMForecaster(hybrid=True)
        >>> # ... (mismo proceso)
    """
    
    def __init__(self, hybrid: bool = False):
        """
        Inicializa el forecaster
        
        Args:
            hybrid: Si True, espera feature 'gp_signal'
        """
        self.hybrid = hybrid
        self.params = {}
        self.model = None
        self.best_iteration = 0
        
        mode = "HÍBRIDO (GP+LGBM)" if hybrid else "PURO"
        logger.info(f"🚀 LGBMForecaster inicializado (modo {mode})")
        
    def optimize(self, 
                X_train: pd.DataFrame, 
                y_train: pd.Series,
                X_val: pd.DataFrame, 
                y_val: pd.Series,
                n_trials: int = 40):
        """
        Busca mejores hiperparámetros con Optuna
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación
            y_val: Target de validación
            n_trials: Número de combinaciones a probar
            
        Returns:
            best_rmse: Mejor RMSE encontrado
            
        Process:
            1. Define espacio de búsqueda
            2. Optuna prueba combinaciones
            3. Cada trial entrena modelo y evalúa
            4. Selecciona mejores parámetros
        """
        logger.info(f"   🔍 Optimizando hiperparámetros ({n_trials} trials)...")
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        def objective(trial):
            """Función objetivo para Optuna"""
            param = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'seed': 42,
                'deterministic': True,
                'force_col_wise': True,
                'feature_pre_filter': False,
                
                # Espacio de búsqueda
                'learning_rate': trial.suggest_float('lr', 0.005, 0.3, log=True),
                'num_leaves': trial.suggest_int('leaves', 20, 100),
                'max_depth': trial.suggest_int('depth', 3, 12),
                'min_data_in_leaf': trial.suggest_int('min_data', 5, 30),
                'lambda_l1': trial.suggest_float('l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('l2', 1e-8, 10.0, log=True),
                'feature_fraction': trial.suggest_float('ff', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bf', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bfq', 1, 7)
            }
            
            # Entrenar con Early Stopping
            model = lgb.train(
                param,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(False)
                ]
            )
            
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))
        
        # Ejecutar optimización
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Guardar mejores parámetros
        self.params = study.best_params
        self.params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'seed': 42,
            'deterministic': True,
            'force_col_wise': True,
            'feature_pre_filter': False
        })
        
        best_rmse = study.best_value
        logger.info(f"   ✅ Optimización completa (Best RMSE={best_rmse:.2f})")
        
        return best_rmse

    def fit(self, 
           X_train: pd.DataFrame, 
           y_train: pd.Series,
           X_val: Optional[pd.DataFrame] = None,
           y_val: Optional[pd.Series] = None):
        """
        Entrena el modelo final
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: (Opcional) Features de validación
            y_val: (Opcional) Target de validación
        """
        logger.info("   🏋️  Entrenando modelo final...")
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = []
        
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets = [dval]
        
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            valid_sets=valid_sets if valid_sets else None,
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(0)
            ] if valid_sets else []
        )
        
        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.model.num_trees()
        
        logger.info(f"   ✅ Modelo entrenado ({self.best_iteration} árboles)")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones
        
        Args:
            X: Features para predecir
            
        Returns:
            Array con predicciones
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta .fit() primero")
        
        return self.model.predict(X, num_iteration=self.best_iteration)


# ============================================================================
# TESTING
# ============================================================================
if __name__ == '__main__':
    print("Módulo lgbm_forecaster.py cargado correctamente")
    print("\nClases disponibles:")
    print("  • LGBMFeatureEngineer")
    print("  • LGBMForecaster (hybrid=False/True)")
