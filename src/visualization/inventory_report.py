# =============================================================================
# MÓDULO: src/visualization/inventory_report.py
# VERSIÓN: 2.0 (Executive Financial Report)
# =============================================================================

"""
Generador de Reportes Financieros de Inventario - Versión Ejecutiva

Mejoras sobre v1.0:
    - Tabla scorecard con heatmap
    - Matriz estratégica 2x2 (BCG-style)
    - Métricas de impacto financiero
    - Layout profesional multi-panel
    
Uso:
    from src.visualization.inventory_report import InventoryReportGenerator
    
    generator = InventoryReportGenerator()
    generator.create_executive_report(
        df_inventory=standard,
        save_path='outputs/reporte_financiero_v2.png'
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryReportGenerator:
    """
    Generador de reportes ejecutivos de inventario
    
    Features:
        - Scorecard table con gradientes
        - Matriz estratégica (Riesgo vs Volumen)
        - Composición de pedidos
        - Impacto financiero estimado
    """
    
    def __init__(self):
        """Inicializa con paleta corporativa"""
        
        # Paleta de colores profesional
        self.colors = {
            'demand': '#3498db',      # Azul - Demanda base
            'safety': '#e74c3c',      # Rojo - Stock seguridad
            'efficient': '#2ecc71',   # Verde - Eficiente
            'moderate': '#f39c12',    # Naranja - Moderado
            'risky': '#c0392b',       # Rojo oscuro - Riesgoso
            'grid': '#ecf0f1',        # Gris claro - Grids
            'text': '#2c3e50'         # Azul oscuro - Texto
        }
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        
    
    def create_executive_report(
        self,
        df_inventory: pd.DataFrame,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """
        Genera reporte ejecutivo completo de 4 paneles
        
        Args:
            df_inventory: DataFrame con columnas:
                - Categoría
                - Modelo
                - Demanda Semanal
                - Stock Seguridad
                - Cantidad Pedido
                - % Riesgo
                - RMSE (opcional)
            save_path: Ruta para guardar imagen
            show_plot: Si mostrar en pantalla
        """
        
        logger.info("🎨 Generando reporte ejecutivo de inventario...")
        
        # Validar datos
        required_cols = ['Categoría', 'Demanda Semanal', 'Stock Seguridad', 
                        'Cantidad Pedido', '% Riesgo']
        if not all(col in df_inventory.columns for col in required_cols):
            raise ValueError(f"Faltan columnas requeridas: {required_cols}")
        
        # Crear figura con grid personalizado
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Scorecard Table (Top-Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_scorecard_table(df_inventory, ax1)
        
        # Panel B: Composición de Pedidos (Top-Right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_order_composition(df_inventory, ax2)
        
        # Panel C: Matriz Estratégica (Bottom-Left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_strategy_matrix(df_inventory, ax3)
        
        # Panel D: Impacto Financiero (Bottom-Right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_financial_impact(df_inventory, ax4)
        
        # Título general
        fig.suptitle(
            '💰 REPORTE EJECUTIVO DE OPTIMIZACIÓN DE INVENTARIO\n'
            'Análisis de Eficiencia y Riesgo por Categoría (Nivel de Servicio 95%)',
            fontsize=18,
            fontweight='bold',
            y=0.98
        )
        
        # Guardar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"✅ Reporte guardado: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    
    def _plot_scorecard_table(self, df: pd.DataFrame, ax):
        """
        Panel A: Tabla scorecard con heatmap
        """
        
        ax.axis('off')
        ax.set_title('📋 SCORECARD DE EFICIENCIA', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Preparar datos para tabla
        table_data = df[['Categoría', 'Demanda Semanal', 'Stock Seguridad', 
                        'Cantidad Pedido', '% Riesgo']].copy()
        
        # Formatear números
        table_data['Demanda Semanal'] = table_data['Demanda Semanal'].apply(lambda x: f'{x:,.0f}')
        table_data['Stock Seguridad'] = table_data['Stock Seguridad'].apply(lambda x: f'{x:,.0f}')
        table_data['Cantidad Pedido'] = table_data['Cantidad Pedido'].apply(lambda x: f'{x:,.0f}')
        table_data['% Riesgo'] = table_data['% Riesgo'].apply(lambda x: f'{x:.1f}%')
        
        # Renombrar columnas para tabla
        table_data.columns = ['Categoría', 'Demanda\nSemanal', 'Stock\nSeguridad', 
                             'Pedido\nTotal', '% Riesgo']
        
        # Crear tabla
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Estilizar
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Colorear header
        for i in range(len(table_data.columns)):
            cell = table[(0, i)]
            cell.set_facecolor(self.colors['text'])
            cell.set_text_props(weight='bold', color='white')
        
        # Colorear filas según % Riesgo
        risk_values = df['% Riesgo'].values
        for i, risk in enumerate(risk_values):
            # Determinar color
            if risk < 20:
                color = self.colors['efficient']
            elif risk < 23:
                color = self.colors['moderate']
            else:
                color = self.colors['risky']
            
            # Aplicar a celda de % Riesgo
            cell = table[(i+1, 4)]
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold', color='white')
            
            # Alternar fondo de otras celdas
            bg_color = '#f8f9fa' if i % 2 == 0 else 'white'
            for j in range(4):
                table[(i+1, j)].set_facecolor(bg_color)
        
        # Leyenda
        legend_elements = [
            mpatches.Patch(color=self.colors['efficient'], label='✓ Bajo Riesgo (<20%)'),
            mpatches.Patch(color=self.colors['moderate'], label='⚠ Moderado (20-23%)'),
            mpatches.Patch(color=self.colors['risky'], label='⚠ Alto Riesgo (>23%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1.15), frameon=False, fontsize=9)
    
    
    def _plot_order_composition(self, df: pd.DataFrame, ax):
        """
        Panel B: Gráfico de composición MEJORADO
        """
        
        # Ordenar por % Riesgo (más claro visualmente)
        df_sorted = df.sort_values('% Riesgo', ascending=True)
        
        categories = df_sorted['Categoría'].values
        demand = df_sorted['Demanda Semanal'].values
        safety = df_sorted['Stock Seguridad'].values
        
        y_pos = np.arange(len(categories))
        
        # Barras apiladas
        ax.barh(y_pos, demand, color=self.colors['demand'], 
               label='Demanda Base', edgecolor='white', linewidth=1.5)
        ax.barh(y_pos, safety, left=demand, color=self.colors['safety'], 
               label='Stock Seguridad', edgecolor='white', linewidth=1.5)
        
        # Etiquetas de % Riesgo
        for i, (cat, risk) in enumerate(zip(categories, df_sorted['% Riesgo'])):
            total = demand[i] + safety[i]
            
            # Color de texto según riesgo
            if risk < 20:
                text_color = self.colors['efficient']
            elif risk < 23:
                text_color = self.colors['moderate']
            else:
                text_color = self.colors['risky']
            
            ax.text(total + 15, i, f'{risk:.1f}%', 
                   va='center', ha='left', fontsize=11,
                   fontweight='bold', color=text_color)
        
        # Configuración
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=10)
        ax.set_xlabel('Unidades', fontsize=12, fontweight='bold')
        ax.set_title('📊 COMPOSICIÓN DEL PEDIDO\nDemanda vs Stock de Seguridad', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Leyenda
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        
        # Grid suave
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    
    def _plot_strategy_matrix(self, df: pd.DataFrame, ax):
        """
        Panel C: Matriz Estratégica BCG-style
        """
        
        x = df['Cantidad Pedido']
        y = df['% Riesgo']
        labels = df['Categoría']
        
        # Determinar colores
        colors = []
        for risk in y:
            if risk < 20:
                colors.append(self.colors['efficient'])
            elif risk < 23:
                colors.append(self.colors['moderate'])
            else:
                colors.append(self.colors['risky'])
        
        # Scatter plot
        scatter = ax.scatter(x, y, s=800, c=colors, alpha=0.7, 
                           edgecolors='black', linewidths=2)
        
        # Líneas de cuadrantes
        x_mean = x.mean()
        y_mean = y.mean()
        
        ax.axvline(x_mean, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(y_mean, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        # Etiquetas de categorías
        for i, label in enumerate(labels):
            # Limpiar nombre
            clean_label = label.replace('_', ' ').title()
            ax.text(x.iloc[i], y.iloc[i], clean_label, 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Anotaciones de cuadrantes
        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        
        # Cuadrante 1: Alto Volumen, Alto Riesgo (Arriba-Derecha)
        ax.text(x_max * 0.95, y_max * 0.95, 
               '⚠️ ALTO RIESGO\nATENCIÓN PRIORITARIA',
               ha='right', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=self.colors['risky'], 
                        alpha=0.2, edgecolor=self.colors['risky']))
        
        # Cuadrante 2: Alto Volumen, Bajo Riesgo (Abajo-Derecha)
        ax.text(x_max * 0.95, y_min * 1.05,
               '✅ CASH COWS\nEFICIENTES',
               ha='right', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=self.colors['efficient'], 
                        alpha=0.2, edgecolor=self.colors['efficient']))
        
        # Cuadrante 3: Bajo Volumen, Alto Riesgo (Arriba-Izquierda)
        ax.text(x_min * 1.05, y_max * 0.95,
               '⚠️ PROBLEMA\nREVISAR PROVEEDORES',
               ha='left', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=self.colors['moderate'], 
                        alpha=0.2, edgecolor=self.colors['moderate']))
        
        # Configuración
        ax.set_xlabel('Volumen de Pedido (Unidades)', fontsize=12, fontweight='bold')
        ax.set_ylabel('% de Riesgo (Stock Seguridad / Total)', fontsize=12, fontweight='bold')
        ax.set_title('🎯 MATRIZ ESTRATÉGICA\nRiesgo vs Volumen', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Grid suave
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    
    def _plot_financial_impact(self, df: pd.DataFrame, ax):
        """
        Panel D: Impacto financiero estimado (NUEVO)
        """
        
        # Calcular "costo de la incertidumbre"
        # Asumimos costo promedio de $50 por unidad de stock de seguridad
        COST_PER_UNIT = 50
        
        df_impact = df.copy()
        df_impact['Costo Riesgo'] = df_impact['Stock Seguridad'] * COST_PER_UNIT
        df_impact = df_impact.sort_values('Costo Riesgo', ascending=True)
        
        categories = df_impact['Categoría'].values
        costs = df_impact['Costo Riesgo'].values
        
        # Barras horizontales
        bars = ax.barh(categories, costs, color=self.colors['risky'], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Colorear según magnitud
        for i, (bar, cost) in enumerate(zip(bars, costs)):
            if cost < costs.mean():
                bar.set_color(self.colors['efficient'])
            elif cost < costs.mean() * 1.5:
                bar.set_color(self.colors['moderate'])
            else:
                bar.set_color(self.colors['risky'])
        
        # Etiquetas de valores
        for i, cost in enumerate(costs):
            ax.text(cost + max(costs)*0.02, i, f'${cost:,.0f}', 
                   va='center', ha='left', fontsize=10, fontweight='bold')
        
        # Total
        total_cost = costs.sum()
        ax.text(0.95, 0.05, 
               f'💰 COSTO TOTAL DE RIESGO:\n${total_cost:,.0f}',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Configuración
        ax.set_xlabel('Costo Estimado de Stock Seguridad ($)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('💵 IMPACTO FINANCIERO\nCosto de Mantener Incertidumbre', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)


# =============================================================================
# FUNCIÓN DE CONVENIENCIA
# =============================================================================

def generate_inventory_report(df_inventory: pd.DataFrame, 
                             save_path: str = 'outputs/reporte_financiero_v2.png'):
    """
    Función rápida para generar reporte
    
    Example:
        >>> from src.visualization.inventory_report import generate_inventory_report
        >>> generate_inventory_report(standard)
    """
    generator = InventoryReportGenerator()
    return generator.create_executive_report(df_inventory, save_path)
