"""
Interfaz Interactiva de Usuario - Olist Logistics Intelligence

Responsabilidad:
    - Dashboard interactivo con widgets
    - Predicción en tiempo real con GP
    - Cálculo dinámico de stock
    - Visualización histórico + proyección

Fase del Proyecto: 2 (Interfaz Final)

Uso:
    from src.visualization.interactive_ui import OlistDashboard
    dashboard = OlistDashboard(df_main)
    dashboard.render()
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from src.models.gaussian_process_model import GaussianProcessForecaster
from src.models.time_series_processor import TimeSeriesProcessor
from config import TIME_SERIES_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OlistDashboard:
    """
    Dashboard interactivo para planificación de inventario
    
    Features:
        - Selección de categoría
        - Ajuste de horizonte temporal
        - Nivel de servicio dinámico
        - Cálculo en tiempo real
        - Visualización integrada
    """
    
    def __init__(self, df_main: pd.DataFrame):
        """
        Args:
            df_main: DataFrame procesado con todas las órdenes
        """
        self.df_main = df_main
        self.model_registry = {}
        self.processor = TimeSeriesProcessor(
            start_date=TIME_SERIES_CONFIG['start_date'],
            end_date=TIME_SERIES_CONFIG['end_date']
        )
        
        # Mapeo de categorías
        self.category_map = {
            'bed_bath_table': 'cama_mesa_banho',
            'health_beauty': 'beleza_saude',
            'sports_leisure': 'esporte_lazer',
            'furniture_decor': 'moveis_decoracao',
            'computers_accessories': 'informatica_acessorios',
            'housewares': 'utilidades_domesticas'
        }
        
        logger.info("OlistDashboard inicializado")
    
    def _build_model_registry(self):
        """Construye registro de modelos entrenados"""
        logger.info("Construyendo registro de modelos...")
        
        # Cargar información de ganadores
        try:
            df_summary = pd.read_csv('outputs/model_selection_summary.csv')

        except FileNotFoundError:
            logger.warning("No se encontró model_selection_summary.csv, usando valores por defecto")
            df_summary = pd.DataFrame({
                'categoria': list(self.category_map.keys()),
                'modelo_seleccionado': ['N/A'] * len(self.category_map),
                'rmse': [0.0] * len(self.category_map)
            })
        
        for cat_en, cat_pt in self.category_map.items():
            try:
                logger.info(f"   Procesando: {cat_en}...")
                
                # Obtener info del ganador
                cat_info = df_summary[df_summary['categoria'] == cat_en]
                if not cat_info.empty:
                    modelo_ganador = cat_info.iloc[0]['modelo_seleccionado']
                    rmse_val = cat_info.iloc[0]['rmse']
                else:
                    modelo_ganador = 'N/A'
                    rmse_val = 0.0
                
                # Procesar datos
                X, y, dates = self.processor.process(self.df_main, category=cat_pt, validate=False)
                
                # Entrenar GP para interfaz
                gp = GaussianProcessForecaster(use_fast_config=False)
                gp.train(X, y)
                
                # Extraer memoria
                insights = gp.get_insights()
                memoria = insights.get('memoria_predictiva', 12.0)
                
                # Crear serie
                y_series = pd.Series(y, index=dates)
                
                # Guardar en registry
                self.model_registry[cat_en] = {
                    'model': gp,
                    'X_train': X,
                    'y_train': y_series,
                    'memory': memoria,
                    'modelo_ganador': modelo_ganador,
                    'rmse': rmse_val
                }
                
                logger.info(f"   ✅ {cat_en}: Memoria={memoria:.1f} sem")
                
            except Exception as e:
                logger.error(f"   ❌ Error en {cat_en}: {e}")
        
        logger.info(f"Registro completo: {len(self.model_registry)} modelos")
    
    def _create_widgets(self):
        """Crea widgets de la interfaz"""
        style = {'description_width': '150px'}
        layout_full = widgets.Layout(width='95%')
        
        # Header
        self.header = widgets.HTML(
            "<div style='background-color:#2C3E50; padding:15px; border-radius:5px; margin-bottom:10px'>"
            "<h2 style='color:white; margin:0; font-family:sans-serif'>🚀 Olist AI: Command Center</h2>"
            "<p style='color:#ecf0f1; margin:5px 0 0 0; font-size:12px'>Sistema Inteligente de Gestión de Inventario</p>"
            "</div>"
        )
        
        # Dropdown categoría
        self.w_cat = widgets.Dropdown(
            options=list(self.model_registry.keys()),
            description='📂 Categoría:',
            style=style,
            layout=layout_full
        )
        
        # Slider horizonte
        self.w_sem = widgets.IntSlider(
            value=12,
            min=4,
            max=52,
            description='📅 Horizonte (Sem):',
            style=style,
            layout=layout_full
        )
        
        # Slider nivel servicio
        self.w_riesgo = widgets.SelectionSlider(
            options=[('Bajo (80%)', 1.28), ('Estándar (95%)', 1.96), ('Crítico (99%)', 2.57)],
            value=1.96,
            description='🛡️ Nivel Servicio:',
            style=style,
            layout=layout_full
        )
        
        # Botón
        self.btn = widgets.Button(
            description=' CALCULAR ESTRATEGIA',
            button_style='info',
            icon='chart-line',
            layout=widgets.Layout(width='95%', height='50px', margin='10px 0')
        )
        
        # Output
        self.out = widgets.Output()
        
        # Vincular eventos
        self.w_cat.observe(self._update_constraints, names='value')
        self.btn.on_click(self._run_calculation)
    
    def _update_constraints(self, change):
        """Ajusta slider según memoria de categoría"""
        if change['type'] == 'change' and change['name'] == 'value':
            cat = change['new']
            if cat in self.model_registry:
                memoria = self.model_registry[cat]['memory']
                nuevo_max = min(int(memoria * 1.5), 52)
                self.w_sem.max = max(8, nuevo_max)
                if self.w_sem.value > self.w_sem.max:
                    self.w_sem.value = self.w_sem.max
    
    def _run_calculation(self, b):
        """Ejecuta cálculo y muestra resultados"""
        with self.out:
            clear_output()
            
            cat = self.w_cat.value
            sem = self.w_sem.value
            z = self.w_riesgo.value
            label_riesgo = [k for k, v in self.w_riesgo.options if v == z][0]
            
            # Recuperar datos
            data = self.model_registry[cat]
            model = data['model']
            y_hist = data['y_train']
            X_hist = data['X_train']
            memoria = data['memory']
            modelo_ganador = data['modelo_ganador']
            rmse = data['rmse']
            
            # Proyección
            n_hist = len(X_hist)
            X_total = np.arange(n_hist + sem).reshape(-1, 1)
            fechas_total = pd.date_range(start=y_hist.index[0], periods=len(X_total), freq='W')
            
            y_pred_total, sigma_total = model.predict(X_total, return_std=True)
            
            y_pred_futuro = y_pred_total[n_hist:]
            sigma_futuro = sigma_total[n_hist:]
            
            # Métricas
            demanda_media = np.sum(y_pred_futuro)
            stock_seguridad = np.sum(sigma_futuro * z)
            total_sugerido = demanda_media + stock_seguridad
            rango_min = np.sum(y_pred_futuro - (sigma_futuro * z))
            rango_max = np.sum(y_pred_futuro + (sigma_futuro * z))
            
            # Display HTML
            display(widgets.HTML(f"""
            <div style="font-family: sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #ffffff; box-shadow: 2px 2px 10px #eee;">
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #3498db;">
                    <span style="font-size: 11px; color: #7f8c8d; font-weight: bold;">ESTRATEGIA RECOMENDADA</span><br>
                    <span style="font-size: 12px;">Modelo Ganador: <b>{modelo_ganador}</b> | RMSE: <b>{rmse:.2f}</b></span>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 15px;">
                    <div>
                        <span style="font-size: 12px; color: #7f8c8d; font-weight: bold;">ORDEN SUGERIDA</span><br>
                        <span style="font-size: 42px; color: #2C3E50; font-weight: 900;">{int(total_sugerido):,} u</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 12px; color: #7f8c8d;">NIVEL SERVICIO</span><br>
                        <span style="font-size: 18px; color: #e74c3c; font-weight: bold;">{label_riesgo}</span>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                    <div style="text-align: center;">
                        <span style="color: #95a5a6; font-size: 11px;">DEMANDA BASE</span><br>
                        <span style="font-size: 18px; font-weight: bold;">{int(demanda_media):,}</span>
                    </div>
                    <div style="text-align: center;">
                        <span style="color: #95a5a6; font-size: 11px;">STOCK SEGURIDAD</span><br>
                        <span style="font-size: 18px; font-weight: bold; color: #e67e22;">+{int(stock_seguridad):,}</span>
                    </div>
                    <div style="text-align: center;">
                        <span style="color: #95a5a6; font-size: 11px;">MEMORIA GP</span><br>
                        <span style="font-size: 18px; font-weight: bold; color: #27ae60;">{memoria:.1f} sem</span>
                    </div>
                </div>
                
                <div style="margin-top: 15px; background: #f8f9fa; padding: 10px; border-radius: 4px; text-align: center;">
                    <span style="font-size: 11px; color: #555;">
                        Rango Probable: <b>{int(rango_min):,}</b> a <b>{int(rango_max):,}</b> unidades
                    </span>
                </div>
            </div>
            """))
            
            # Gráfico
            plt.figure(figsize=(14, 6))
            plt.plot(fechas_total[:n_hist], y_hist, 'k.', alpha=0.3, markersize=8, label='Ventas Reales')
            plt.plot(fechas_total, y_pred_total, color='#2980b9', linewidth=2.5, label='Predicción GP')
            plt.fill_between(fechas_total, y_pred_total - (sigma_total * z), y_pred_total + (sigma_total * z),
                           color='#3498db', alpha=0.2, label=f'Zona de Incertidumbre ({label_riesgo})')
            plt.axvline(x=fechas_total[n_hist-1], color='#c0392b', linestyle='--', linewidth=2, label='Hoy')
            plt.title(f"Planificación Integral: {cat.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
            plt.ylabel("Volumen Semanal", fontsize=11)
            plt.xlabel("Línea de Tiempo", fontsize=11)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def render(self):
        """Renderiza la interfaz completa"""
        logger.info("Construyendo interfaz...")
        
        # Construir registro
        self._build_model_registry()
        
        # Crear widgets
        self._create_widgets()
        
        # Inicializar constraints
        if len(self.model_registry) > 0:
            first_cat = list(self.model_registry.keys())[0]
            self._update_constraints({'type': 'change', 'name': 'value', 'new': first_cat})
        
        # Mostrar
        logger.info("✅ Interfaz lista")
        display(widgets.VBox([self.header, self.w_cat, self.w_sem, self.w_riesgo, self.btn, self.out]))


if __name__ == '__main__':
    print("Módulo interactive_ui.py cargado")
