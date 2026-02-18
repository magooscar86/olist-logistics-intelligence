"""
Módulo de Visualización de Torneo - Olist Logistics Intelligence

Responsabilidad:
    - Tabla ejecutiva con colores
    - Dashboard de ganadores
    - Heatmap de comparación
    - Gráficos para presentaciones

Fase del Proyecto: 2 (Visualización Final)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TournamentVisualizer:
    """
    Generador de visualizaciones ejecutivas
    """
    
    def __init__(self):
        self.colors = {
            'winner': '#2ecc71',      # Verde brillante
            'second': '#f39c12',      # Naranja
            'third': '#3498db',       # Azul
            'poor': '#e74c3c',        # Rojo
            'baseline': '#95a5a6',    # Gris
            'gp': '#9b59b6',          # Morado
            'lgbm': '#16a085',        # Verde azulado
            'hybrid': '#e67e22'       # Naranja oscuro
        }
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
    
    def create_winners_dashboard(self, results_df: pd.DataFrame,
                                save_path: Optional[str] = None):
        """
        Dashboard completo con 4 paneles (VERSIÓN 3.7 - RIGOR CIENTÍFICO)

        Paneles:
            A. Ranking de Precisión (RMSE)
            B. Valor Agregado (% Mejora vs Baseline)
            C. Mapa de Consistencia (Precisión vs Estabilidad) - Solo modelos CV
            D. Distribución de Estrategias (Donut Chart)

        Args:
            results_df: DataFrame con resultados del torneo
            save_path: Ruta para guardar (opcional)

        Cambios vs versión anterior:
            - Scatter plot muestra solo modelos con CV (rigor científico)
            - Zona de alto rendimiento marcada
            - Mejora vs baseline con etiquetas claras
            - Donut chart con número de categorías en centro
        """
        logger.info("🎨 Generando Dashboard (Rigor Científico)...")

        # --- 1. PROCESAMIENTO ---
        # Detectar ganadores
        idx_winners = results_df.groupby('Categoría')['RMSE_mean'].idxmin()
        df_viz = results_df.loc[idx_winners].copy()

        # Renombrar columnas
        df_viz.rename(columns={'RMSE_mean': 'RMSE', 'RMSE_std': 'RMSE_std', 'Modelo': 'Ganador'}, inplace=True)

        # --- 2. CÁLCULO DE MEJORA ---
        baseline_rmse = results_df[results_df['Modelo'] == 'Media Móvil'].set_index('Categoría')['RMSE_mean']
        df_viz['Mejora'] = df_viz.apply(
            lambda x: ((baseline_rmse.get(x['Categoría'], x['RMSE']) - x['RMSE']) / baseline_rmse.get(x['Categoría'], x['RMSE'])) * 100,
            axis=1
        )

        # --- 3. CONFIGURACIÓN VISUAL ---
        short_names = {
            'cama_mesa_banho': 'Bed & Bath', 'bed_bath_table': 'Bed & Bath',
            'beleza_saude': 'Health', 'health_beauty': 'Health',
            'esporte_lazer': 'Sports', 'sports_leisure': 'Sports',
            'moveis_decoracao': 'Furniture', 'furniture_decor': 'Furniture',
            'informatica_acessorios': 'Computers', 'computers_accessories': 'Computers',
            'utilidades_domesticas': 'Housewares', 'housewares': 'Housewares'
        }
        df_viz['ShortName'] = df_viz['Categoría'].map(short_names).fillna(df_viz['Categoría'])

        # Paleta de colores
        colors = {
            'Media Móvil': '#95A5A6',
            'Holt-Winters': '#F39C12',
            'GP Puro': '#3498DB',
            'LightGBM Puro': '#2ECC71',
            'Híbrido (GP+LGBM)': '#9B59B6'
        }

        # --- 4. GENERACIÓN DE GRÁFICOS ---
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))

        # ============================================================
        # PANEL A: RANKING DE PRECISIÓN
        # ============================================================
        ax1 = axes[0, 0]
        df_sorted = df_viz.sort_values('RMSE')
        sns.barplot(data=df_sorted, x='RMSE', y='ShortName', hue='Ganador',
                   palette=colors, dodge=False, ax=ax1, legend=True)

        # Añadir valores
        for i, val in enumerate(df_sorted['RMSE']):
            ax1.text(val + 0.5, i, f"{val:.1f}", va='center', fontweight='bold', color='#333')

        ax1.set_title('A. Precisión (RMSE)', fontweight='bold', loc='left', fontsize=13)
        ax1.set_xlabel('RMSE (menor = mejor)', fontweight='bold')
        ax1.set_ylabel('')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='x')

        # ============================================================
        # PANEL B: VALOR AGREGADO
        # ============================================================
        ax2 = axes[0, 1]
        df_imp = df_viz.sort_values('Mejora', ascending=False)
        bar_colors = ['#27AE60' if x > 1 else '#BDC3C7' for x in df_imp['Mejora']]

        sns.barplot(data=df_imp, x='Mejora', y='ShortName', ax=ax2,
                   palette=bar_colors, hue='ShortName', legend=False)

        # Añadir etiquetas
        for i, val in enumerate(df_imp['Mejora']):
            label = f"+{val:.1f}%" if val > 1 else "Baseline"
            x_pos = val + 0.5 if val > 0 else 0.5
            color = 'black' if val > 1 else 'gray'
            ax2.text(x_pos, i, label, va='center', fontweight='bold', color=color)

        ax2.set_title('B. Valor Agregado (% Mejora)', fontweight='bold', loc='left', fontsize=13)
        ax2.set_xlabel('Mejora vs Baseline (%)', fontweight='bold')
        ax2.set_ylabel('')
        ax2.axvline(0, color='black', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')

        # ============================================================
        # PANEL C: MAPA DE CONSISTENCIA (SOLO MODELOS CON CV)
        # ============================================================
        ax3 = axes[1, 0]

        # FILTRO DE RIGOR: Solo modelos con desviación estándar válida
        df_scatter = df_viz.dropna(subset=['RMSE_std']).copy()

        if not df_scatter.empty:
            sns.scatterplot(data=df_scatter, x='RMSE', y='RMSE_std', hue='Ganador',
                           palette=colors, s=600, alpha=0.8, ax=ax3,
                           edgecolor='black', linewidth=2, zorder=5, legend=False)

            # Etiquetas con offsets
            offsets = {
                'Furniture': (0, 0.5),
                'Health': (0, -0.5),
                'Computers': (-2, 0)
            }
            for i, row in df_scatter.iterrows():
                name = row['ShortName']
                off_x, off_y = offsets.get(name, (0, 0.5))
                ax3.text(row['RMSE'] + off_x, row['RMSE_std'] + off_y, name,
                        ha='center', fontweight='bold', color='#2C3E50', fontsize=10)

            # Zona de Alto Rendimiento
            xlim = ax3.get_xlim()
            ylim = ax3.get_ylim()
            rect = plt.Rectangle((xlim[0], ylim[0]),
                                (xlim[1]-xlim[0])/3, (ylim[1]-ylim[0])/3,
                                color='green', alpha=0.05, zorder=0)
            ax3.add_patch(rect)
            ax3.text(xlim[0]+1, ylim[0]+0.2, 'Zona de Alto Rendimiento',
                    color='green', fontweight='bold', fontsize=9)

            logger.info(f"ℹ️ Scatter plot: {len(df_scatter)}/{len(df_viz)} modelos con CV")
        else:
            ax3.text(0.5, 0.5, "No hay modelos con validación cruzada (CV) ganadores.",
                    ha='center', va='center', transform=ax3.transAxes, fontsize=11)

        ax3.set_title('C. Mapa de Consistencia (Solo modelos con CV)',
                     fontweight='bold', loc='left', fontsize=13)
        ax3.set_xlabel('Error Promedio (RMSE)', fontweight='bold')
        ax3.set_ylabel('Inestabilidad (Desviación Estándar)', fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.3)

        # ============================================================
        # PANEL D: DISTRIBUCIÓN DE ESTRATEGIAS (DONUT)
        # ============================================================
        ax4 = axes[1, 1]
        win_counts = df_viz['Ganador'].value_counts()
        pie_cols = [colors.get(x, '#333') for x in win_counts.index]

        wedges, texts, autotexts = ax4.pie(
            win_counts,
            labels=win_counts.index,
            autopct='%1.0f%%',
            startangle=90,
            colors=pie_cols,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            textprops={'fontsize': 10}
        )

        # Texto en centro
        plt.setp(autotexts, size=11, weight="bold", color="white")
        ax4.text(0, 0, f"{len(df_viz)}\nCategorías",
                ha='center', va='center', fontweight='bold',
                color='#555', fontsize=12)

        ax4.set_title('D. Distribución de Estrategias',
                     fontweight='bold', loc='left', fontsize=13)

        # Título general
        fig.suptitle('Dashboard Ejecutivo: Torneo de Modelos (v3.7)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        # Guardar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Dashboard guardado: {save_path}")

        plt.show()

        # Resumen de exclusiones
        excluded = len(df_viz) - len(df_scatter)
        if excluded > 0:
            logger.info(f"ℹ️ Se excluyeron {excluded} categorías del Panel C por no tener CV")

        # Texto en centro
        plt.setp(autotexts, size=11, weight="bold", color="white")
        ax4.text(0, 0, f"{len(df_viz)}\nCategorías", 
                ha='center', va='center', fontweight='bold', 
                color='#555', fontsize=12)
        
        ax4.set_title('D. Distribución de Estrategias', 
                     fontweight='bold', loc='left', fontsize=13)
        
        # Título general
        fig.suptitle('Dashboard Ejecutivo: Torneo de Modelos (v3.7)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Guardar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Dashboard guardado: {save_path}")
        
        plt.show()
        
        # Resumen de exclusiones
        excluded = len(df_viz) - len(df_scatter)
        if excluded > 0:
            logger.info(f"ℹ️ Se excluyeron {excluded} categorías del Panel C por no tener CV")

    def create_heatmap(self, results_df: pd.DataFrame, 
                      save_path: Optional[str] = None):
        """
        Heatmap de comparación de todos los modelos
        
        Args:
            results_df: DataFrame con resultados
            save_path: Ruta para guardar
        """
        pivot = results_df.pivot(index='Categoría', columns='Modelo', values='RMSE_mean')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'RMSE'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Heatmap: Comparación Completa de Modelos', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
        ax.set_ylabel('Categoría', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap guardado: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_table(self, summary_df: pd.DataFrame):
        """
        Tabla resumen con formato HTML para Jupyter
        
        Args:
            summary_df: DataFrame del selector automático
            
        Returns:
            Styled DataFrame
        """
        display_df = summary_df[[
            'categoria', 'modelo_seleccionado', 'rmse', 
            'estabilidad', 'mejora_vs_baseline', 'nivel_confianza'
        ]].copy()
        
        display_df.columns = [
            'Categoría', 'Modelo', 'RMSE', 
            'Estabilidad', 'Mejora (%)', 'Confianza'
        ]
        
        def color_mejora(val):
            if val > 20:
                return 'background-color: #2ecc71; color: white'
            elif val > 10:
                return 'background-color: #f39c12; color: white'
            elif val > 0:
                return 'background-color: #3498db; color: white'
            else:
                return 'background-color: #e74c3c; color: white'
        
        def color_confianza(val):
            if val == 'ALTO':
                return 'background-color: #2ecc71; color: white; font-weight: bold'
            elif val == 'MEDIO':
                return 'background-color: #f39c12; color: white'
            else:
                return 'background-color: #e74c3c; color: white'
        
        styled = display_df.style\
            .applymap(color_mejora, subset=['Mejora (%)'])\
            .applymap(color_confianza, subset=['Confianza'])\
            .format({'RMSE': '{:.2f}', 'Mejora (%)': '{:+.1f}%'})\
            .set_caption("Resumen Ejecutivo: Modelos Seleccionados")\
            .set_table_styles([
                {'selector': 'caption', 'props': [('font-size', '14pt'), ('font-weight', 'bold')]},
                {'selector': 'th', 'props': [('background-color', '#34495e'), ('color', 'white'), ('font-weight', 'bold')]}
            ])
        
        return styled


if __name__ == '__main__':
    print("Módulo tournament_viz.py cargado")
