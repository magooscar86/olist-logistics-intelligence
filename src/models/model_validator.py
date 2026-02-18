"""
Módulo de Validación y Benchmark - Olist Logistics Intelligence

Responsabilidad:
    - Orquestar torneo de modelos con reglas justas
    - Garantizar split temporal estricto (NO data leakage)
    - Entrenar todos los modelos con mismos datos
    - Calcular métricas estandarizadas (RMSE, MAE, MAPE)
    - Generar leaderboard y visualizaciones

Fase del Proyecto: 2 (Evaluación Final)

Conceptos Clave:
    - Data Leakage: Cuando el modelo "ve" el futuro durante entrenamiento
    - Split Temporal: Dividir datos por fecha (no aleatorio)
    - Offset: Alineación de índices tras crear features con lags
    - Leaderboard: Tabla comparativa de todos los modelos
    
Modelos que compiten:
    1. Media Móvil (baseline simple)
    2. Holt-Winters (baseline industry)
    3. GP Puro (nuestro modelo core)
    4. LightGBM Puro (machine learning)
    5. Híbrido GP+LGBM (lo mejor de ambos mundos)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple

# Importar nuestros modelos
from src.models.baselines import MovingAverageModel, HoltWintersModel
from src.models.gaussian_process_model import GaussianProcessForecaster
from src.models.lgbm_forecaster import LGBMFeatureEngineer, LGBMForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Árbitro imparcial del torneo de modelos
    
    Garantiza:
        - Todos los modelos usan exactamente los mismos datos
        - Split temporal estricto (sin data leakage)
        - Métricas calculadas de forma consistente
        - Resultados reproducibles
        
    Example:
        >>> validator = ModelValidator(test_weeks=12)
        >>> validator.run_benchmark(df_ts, 'bed_bath_table')
        >>> leaderboard = validator.get_leaderboard()
        >>> validator.plot_results()
    """
    
    def __init__(self, test_weeks: int = 12):
        """
        Inicializa el validator
        
        Args:
            test_weeks: Número de semanas para test (holdout)
        """
        self.test_weeks = test_weeks
        self.results = []
        
        logger.info(f"ModelValidator inicializado (test_weeks={test_weeks})")
    
    def run_benchmark(self,
                     df_ts: pd.DataFrame,
                     category_name: str,
                     use_fast_config: bool = False) -> Dict:
        """
        Ejecuta el torneo completo para una categoría
        
        Args:
            df_ts: DataFrame con columna 'ventas' e índice datetime
            category_name: Nombre de la categoría (para logging)
            use_fast_config: Si True, usa GP_KERNEL_CONFIG_FAST
            
        Returns:
            Dict con métricas por modelo
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"BENCHMARK: {category_name}")
        logger.info(f"{'='*70}")
        
        # Split temporal
        logger.info("\n[1/7] Split temporal...")
        split_idx_raw = len(df_ts) - self.test_weeks
        logger.info(f"   Train: {split_idx_raw} semanas | Test: {self.test_weeks} semanas")
        
        # Entrenar GP (solo con pasado)
        logger.info("\n[2/7] Entrenando GP (sin ver futuro)...")
        X_time = np.arange(len(df_ts)).reshape(-1, 1)
        X_time_train = X_time[:split_idx_raw]
        y_train_raw = df_ts['ventas'].iloc[:split_idx_raw]
        
        gp = GaussianProcessForecaster(use_fast_config=use_fast_config)
        gp.train(X_time_train, y_train_raw.values)
        
        # Generar GP signal
        logger.info("\n[3/7] Generando GP signal...")
        gp_signal_full, _ = gp.predict(X_time, return_std=True)
        logger.info(f"   GP signal: {len(gp_signal_full)} valores")
        
        # Feature Engineering
        logger.info("\n[4/7] Feature Engineering...")
        fe = LGBMFeatureEngineer()
        df_pure = fe.create_features(df_ts, gp_signal=None)
        df_hybrid = fe.create_features(df_ts, gp_signal=gp_signal_full)
        
        offset = len(df_ts) - len(df_pure)
        logger.info(f"   Offset detectado: {offset} filas (por lags)")
        
        split_idx = len(df_pure) - self.test_weeks
        
        y = df_pure['ventas']
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        X_pure = df_pure.drop('ventas', axis=1)
        X_hybrid = df_hybrid.drop('ventas', axis=1)
        
        X_train_pure = X_pure.iloc[:split_idx]
        X_test_pure = X_pure.iloc[split_idx:]
        X_train_hyb = X_hybrid.iloc[:split_idx]
        X_test_hyb = X_hybrid.iloc[split_idx:]
        
        val_size = 8
        train_val_idx = len(y_train) - val_size
        
        y_tr_opt = y_train.iloc[:train_val_idx]
        y_val_opt = y_train.iloc[train_val_idx:]
        X_tr_pure_opt = X_train_pure.iloc[:train_val_idx]
        X_val_pure_opt = X_train_pure.iloc[train_val_idx:]
        X_tr_hyb_opt = X_train_hyb.iloc[:train_val_idx]
        X_val_hyb_opt = X_train_hyb.iloc[train_val_idx:]
        
        logger.info(f"   Train_opt: {len(y_tr_opt)} | Val_opt: {val_size} | Test: {len(y_test)}")
        
        # Entrenar modelos
        logger.info("\n[5/7] Entrenando modelos...")
        
        logger.info("\n   [1/5] Media Móvil...")
        ma = MovingAverageModel()
        ma.optimize(y_tr_opt, y_val_opt, n_trials=20)
        ma.fit(y_train)
        preds_ma = ma.predict_walk_forward(y_test)
        
        logger.info("\n   [2/5] Holt-Winters...")
        holt = HoltWintersModel()
        holt.optimize(y_tr_opt, y_val_opt, n_trials=10)
        holt.fit(y_train)
        preds_holt = holt.predict(len(y_test))
        
        logger.info("\n   [3/5] GP Puro...")
        preds_gp = gp_signal_full[offset + split_idx:]
        
        if len(preds_gp) != len(y_test):
            logger.error(f"   GP length mismatch: {len(preds_gp)} vs {len(y_test)}")
            preds_gp = np.zeros(len(y_test))
        
        logger.info("\n   [4/5] LightGBM Puro...")
        lgb_pure = LGBMForecaster(hybrid=False)
        lgb_pure.optimize(X_tr_pure_opt, y_tr_opt, X_val_pure_opt, y_val_opt, n_trials=40)
        lgb_pure.fit(X_train_pure, y_train, X_val_pure_opt, y_val_opt)
        preds_lgb = lgb_pure.predict(X_test_pure)
        
        logger.info("\n   [5/5] Híbrido (GP+LGBM)...")
        lgb_hyb = LGBMForecaster(hybrid=True)
        lgb_hyb.optimize(X_tr_hyb_opt, y_tr_opt, X_val_hyb_opt, y_val_opt, n_trials=40)
        lgb_hyb.fit(X_train_hyb, y_train, X_val_hyb_opt, y_val_opt)
        preds_hyb = lgb_hyb.predict(X_test_hyb)
        
        # Calcular métricas
        logger.info("\n[6/7] Calculando métricas...")
        
        models = {
            'Media Móvil': preds_ma,
            'Holt-Winters': preds_holt,
            'GP Puro': preds_gp,
            'LightGBM Puro': preds_lgb,
            'Híbrido (GP+LGBM)': preds_hyb
        }
        
        metrics_dict = {}
        
        for name, preds in models.items():
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            
            try:
                mape = mean_absolute_percentage_error(y_test, preds) * 100
            except:
                mape = np.inf
            
            metrics_dict[name] = rmse
            
            self.results.append({
                'Categoría': category_name,
                'Modelo': name,
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'MAPE (%)': round(mape, 2)
            })
        
        # Determinar ganador
        logger.info("\n[7/7] Resultados...")
        
        best_model = min(metrics_dict, key=metrics_dict.get)
        best_rmse = metrics_dict[best_model]
        
        logger.info(f"\nGanador: {best_model}")
        logger.info(f"   RMSE: {best_rmse:.2f}")
        
        logger.info(f"\nTabla de métricas:")
        for name, rmse in sorted(metrics_dict.items(), key=lambda x: x[1]):
            logger.info(f"   {name:20s}: RMSE={rmse:.2f}")
        
        return metrics_dict
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Genera tabla resumen con todos los resultados"""
        if not self.results:
            logger.warning("No hay resultados. Ejecuta run_benchmark() primero")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df = df.sort_values(['Categoría', 'RMSE'])
        
        return df
    
    def plot_results(self, save_path: Optional[str] = None):
        """Genera gráfico comparativo de modelos"""
        if not self.results:
            logger.warning("No hay resultados para graficar")
            return
        
        df = self.get_leaderboard()
        df_pivot = df.pivot(index='Categoría', columns='Modelo', values='RMSE')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        df_pivot.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Benchmark de Modelos por Categoría', fontsize=16, fontweight='bold')
        ax.set_xlabel('Categoría', fontsize=12)
        ax.set_ylabel('RMSE (menor = mejor)', fontsize=12)
        ax.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {save_path}")
        
        plt.show()


if __name__ == '__main__':
    print("Módulo model_validator.py cargado correctamente")
