"""
Módulo de Visualización - Forecasting

Responsabilidad:
    - Gráficos de demanda histórica
    - Predicciones con bandas de incertidumbre
    - Backtesting visual
    - Comparaciones entre modelos

Estilo: Coherente con reportes ejecutivos (Clean Business Style)
Fase del Proyecto: 2 (Forecasting)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import logging
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingVisualizer:
    """
    Clase para visualizaciones de forecasting con GP.
    Diseñada para generar gráficos listos para presentación ejecutiva.
    
    Example:
        >>> viz = ForecastingVisualizer()
        >>> viz.plot_forecast(dates, y_hist, y_pred, sigma, split_point=75)
    """
    
    def __init__(self, figsize=(14, 7)):
        """
        Inicializa el visualizador con el estilo del proyecto
        """
        self.figsize = figsize
        
        # Configurar estilo global
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 12
        
        # Paleta de Colores Corporativa (Olist Project)
        self.colors = {
            'historical': '#2C3E50',    # Azul oscuro (Datos reales)
            'prediction': '#2980B9',    # Azul brillante (Modelo)
            'uncertainty': '#3498DB',   # Azul claro (Sombra)
            'validation': '#E74C3C',    # Rojo (Test/Futuro Real)
            'split': '#7F8C8D',         # Gris (Línea de corte)
            'ma': '#95A5A6',            # Gris claro (Media Móvil)
            'holt': '#F39C12',          # Naranja (Holt)
            'lgbm': '#27AE60'           # Verde (LightGBM)
        }
    
    def plot_historical_demand(self,
                               dates: pd.DatetimeIndex,
                               y: np.ndarray,
                               title: str = "Demanda Histórica Semanal",
                               ylabel: str = "Cantidad de Órdenes",
                               save_path: Optional[str] = None):
        """Gráfico simple de la serie temporal"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, y, 'o-', 
                color=self.colors['historical'],
                markersize=5,
                linewidth=1,
                label='Ventas Reales',
                alpha=0.8)
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Fecha')
        ax.legend(loc='upper left', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_forecast(self,
                     dates_hist: pd.DatetimeIndex,
                     y_hist: np.ndarray,
                     dates_full: pd.DatetimeIndex,
                     y_pred: np.ndarray,
                     sigma: np.ndarray,
                     n_historical: int,
                     confidence: float = 0.95,
                     title: str = "Pronóstico de Demanda (GP)",
                     save_path: Optional[str] = None):
        """Gráfico de predicción con bandas de confianza"""
        fig, ax = plt.subplots(figsize=self.figsize)
        z_score = 1.96 if confidence == 0.95 else 1.28
        
        # 1. Historia
        ax.plot(dates_hist, y_hist, 'k.', 
                markersize=6, alpha=0.4, label='Historia')
        
        # 2. Predicción (Línea Continua)
        ax.plot(dates_full, y_pred, 
                color=self.colors['prediction'], 
                linewidth=2, label='Tendencia (Modelo)')
        
        # 3. Incertidumbre (Solo futuro o todo, aquí todo para ver ajuste)
        ax.fill_between(
            dates_full,
            y_pred - z_score * sigma,
            y_pred + z_score * sigma,
            color=self.colors['uncertainty'],
            alpha=0.15,
            label=f'Riesgo ({int(confidence*100)}%)'
        )
        
        # 4. Línea "HOY"
        cutoff_date = dates_hist[-1]
        ax.axvline(x=cutoff_date, color=self.colors['split'], 
                  linestyle='--', linewidth=1.5)
        ax.text(cutoff_date, y_pred.max(), '  HOY', 
               color=self.colors['split'], fontweight='bold')
        
        ax.set_title(title)
        ax.set_ylabel('Volumen Semanal')
        ax.legend(loc='upper left', shadow=True)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_backtesting(self,
                        dates: pd.DatetimeIndex,
                        y: np.ndarray,
                        y_pred: np.ndarray,
                        sigma: np.ndarray,
                        split_point: int,
                        rmse: float,
                        mae: float,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None):
        """Gráfico de validación retrospectiva (Training vs Test)"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        dates_train = dates[:split_point]
        dates_test = dates[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        # 1. Entrenamiento (Negro)
        ax.plot(dates_train, y_train, 'k.', markersize=6, alpha=0.3, label='Entrenamiento')
        
        # 2. Test (Rojo)
        ax.plot(dates_test, y_test, 'o', color=self.colors['validation'], 
                markersize=6, label='Test (Oculto)')
        
        # 3. Modelo (Azul)
        ax.plot(dates, y_pred, color=self.colors['prediction'], 
                linewidth=2, label='Modelo Validado')
        
        # 4. Sombra
        ax.fill_between(dates, 
                       y_pred - 1.96*sigma, y_pred + 1.96*sigma,
                       color=self.colors['uncertainty'], alpha=0.15)
        
        # 5. Corte
        ax.axvline(x=dates[split_point], color=self.colors['split'], linestyle='--')
        
        # Título métrico
        if title is None:
            mape = (mae / y_test.mean()) * 100
            title = f'Backtesting Robusto: RMSE {rmse:.1f} | MAPE {mape:.1f}%'
            
        ax.set_title(title)
        ax.legend(loc='upper left', shadow=True)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()
        
    def plot_tournament(self,
                       df_results: pd.DataFrame,
                       title: str = "Torneo de Modelos por Categoría",
                       metric: str = "RMSE",
                       save_path: Optional[str] = None):
        """
        Gráfico de barras para comparar modelos (Benchmark)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Preparar datos (Melt)
        # Asume columnas como 'RMSE MA', 'RMSE Holt', etc.
        cols = [c for c in df_results.columns if metric in c]
        df_melt = df_results.melt(id_vars='Categoría', value_vars=cols, 
                                 var_name='Modelo', value_name='Error')
        
        # Limpiar nombres
        df_melt['Modelo'] = df_melt['Modelo'].str.replace(f'{metric} ', '')
        
        # Barplot
        palette = {
            'MA': self.colors['ma'],
            'Holt': self.colors['holt'],
            'GP': self.colors['prediction'],
            'LGBM': self.colors['lgbm'],
            'Híbrido': '#8E44AD'
        }
        
        sns.barplot(data=df_melt, x='Categoría', y='Error', hue='Modelo', 
                   palette=palette, ax=ax, edgecolor='white')
        
        ax.set_title(f"{title} ({metric})")
        ax.set_ylabel(metric)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        if save_path: plt.savefig(save_path, dpi=300)
        plt.show()

if __name__ == '__main__':
    print("Módulo forecasting_viz.py cargado correctamente")
