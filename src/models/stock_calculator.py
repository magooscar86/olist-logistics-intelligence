"""
Calculadora de Stock Óptimo - Olist Logistics Intelligence

Responsabilidad:
    - Usar predicciones del modelo ganador
    - Calcular stock de seguridad
    - Niveles de servicio (80%, 95%, 99%)
    - Generar órdenes de compra

Fase del Proyecto: 2 (Aplicación de Negocio)

Fórmula:
    Stock Óptimo = Demanda Esperada + Stock Seguridad
    Stock Seguridad = Z-score × RMSE × √(Lead Time)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockCalculator:
    """
    Calculadora de inventario óptimo basada en predicciones
    """
    
    # Z-scores para niveles de servicio
    Z_SCORES = {
        '80%': 0.84,   # 80% probabilidad de no desabastecer
        '90%': 1.28,
        '95%': 1.96,   # Estándar en retail
        '99%': 2.57    # Muy conservador
    }
    
    def __init__(self, results_df: pd.DataFrame, summary_df: pd.DataFrame):
        """
        Args:
            results_df: Resultados del torneo (con RMSE)
            summary_df: Resumen del selector (modelos ganadores)
        """
        self.results = results_df
        self.summary = summary_df
        
        logger.info("StockCalculator inicializado")
    
    def calculate_safety_stock(self, 
                               rmse: float, 
                               lead_time_weeks: int = 2,
                               service_level: str = '95%') -> float:
        """
        Calcula stock de seguridad
        
        Args:
            rmse: Error del modelo (desviación esperada)
            lead_time_weeks: Tiempo de reposición en semanas
            service_level: '80%', '90%', '95%', '99%'
            
        Returns:
            Unidades de stock de seguridad
            
        Formula:
            SS = Z × σ × √LT
            
            Donde:
            - Z = Z-score del nivel de servicio
            - σ = RMSE (desviación de demanda)
            - LT = Lead time
        """
        z = self.Z_SCORES[service_level]
        safety_stock = z * rmse * np.sqrt(lead_time_weeks)
        
        return safety_stock
    
    def calculate_reorder_point(self,
                               avg_demand: float,
                               lead_time_weeks: int,
                               safety_stock: float) -> float:
        """
        Punto de reorden (cuándo hacer pedido)
        
        Args:
            avg_demand: Demanda promedio semanal
            lead_time_weeks: Tiempo de reposición
            safety_stock: Stock de seguridad
            
        Returns:
            Punto de reorden en unidades
            
        Formula:
            ROP = (Demanda Semanal × Lead Time) + SS
        """
        lead_time_demand = avg_demand * lead_time_weeks
        reorder_point = lead_time_demand + safety_stock
        
        return reorder_point
    
    def calculate_order_quantity(self,
                                avg_demand: float,
                                review_period_weeks: int = 4,
                                safety_stock: float = 0) -> float:
        """
        Cantidad de pedido óptima
        
        Args:
            avg_demand: Demanda promedio semanal
            review_period_weeks: Cada cuánto revisas inventario
            safety_stock: Stock de seguridad
            
        Returns:
            Cantidad a pedir
            
        Formula:
            Q = (Demanda × Review Period) + SS
        """
        order_qty = (avg_demand * review_period_weeks) + safety_stock
        
        return order_qty
    
    def generate_inventory_plan(self,
                                category: str,
                                avg_weekly_demand: float,
                                lead_time_weeks: int = 2,
                                review_period_weeks: int = 4,
                                service_levels: List[str] = ['80%', '95%', '99%']) -> pd.DataFrame:
        """
        Genera plan de inventario completo para una categoría
        
        Args:
            category: Nombre de categoría
            avg_weekly_demand: Demanda promedio semanal
            lead_time_weeks: Tiempo de reposición
            review_period_weeks: Frecuencia de revisión
            service_levels: Niveles de servicio a calcular
            
        Returns:
            DataFrame con plan por nivel de servicio
        """
        # Obtener RMSE del modelo ganador
        cat_summary = self.summary[self.summary['categoria'] == category]
        
        if cat_summary.empty:
            raise ValueError(f"Categoría {category} no encontrada")
        
        rmse = cat_summary.iloc[0]['rmse']
        modelo = cat_summary.iloc[0]['modelo_seleccionado']
        
        plans = []
        
        for service_level in service_levels:
            # Calcular componentes
            ss = self.calculate_safety_stock(rmse, lead_time_weeks, service_level)
            rop = self.calculate_reorder_point(avg_weekly_demand, lead_time_weeks, ss)
            order_qty = self.calculate_order_quantity(avg_weekly_demand, review_period_weeks, ss)
            
            # Calcular costos (ejemplo)
            avg_inventory = order_qty / 2 + ss  # Inventario promedio
            
            plans.append({
                'Categoría': category,
                'Modelo': modelo,
                'Nivel Servicio': service_level,
                'RMSE': rmse,
                'Demanda Semanal': avg_weekly_demand,
                'Stock Seguridad': np.ceil(ss),
                'Punto Reorden': np.ceil(rop),
                'Cantidad Pedido': np.ceil(order_qty),
                'Inventario Promedio': np.ceil(avg_inventory)
            })
        
        return pd.DataFrame(plans)
    
    def generate_all_plans(self,
                          demand_data: Dict[str, float],
                          lead_time_weeks: int = 2,
                          review_period_weeks: int = 4) -> pd.DataFrame:
        """
        Genera planes para todas las categorías
        
        Args:
            demand_data: Dict {categoria: demanda_semanal_promedio}
            lead_time_weeks: Tiempo de reposición
            review_period_weeks: Frecuencia de revisión
            
        Returns:
            DataFrame con todos los planes
        """
        all_plans = []
        
        for category, avg_demand in demand_data.items():
            try:
                plan = self.generate_inventory_plan(
                    category, 
                    avg_demand, 
                    lead_time_weeks,
                    review_period_weeks,
                    service_levels=['80%', '95%', '99%']
                )
                all_plans.append(plan)
            except Exception as e:
                logger.error(f"Error en {category}: {e}")
        
        if all_plans:
            return pd.concat(all_plans, ignore_index=True)
        return pd.DataFrame()
    
    def get_executive_summary(self, plans_df: pd.DataFrame) -> str:
        """
        Genera reporte ejecutivo de inventario
        
        Args:
            plans_df: DataFrame con planes de inventario
            
        Returns:
            String con reporte formateado
        """
        lines = []
        lines.append("="*70)
        lines.append("REPORTE EJECUTIVO: PLAN DE INVENTARIO")
        lines.append("="*70)
        lines.append("")
        
        # Recomendación estándar (95%)
        standard_plans = plans_df[plans_df['Nivel Servicio'] == '95%']
        
        for _, row in standard_plans.iterrows():
            lines.append(f"[{row['Categoría'].upper()}]")
            lines.append(f"  Modelo usado:        {row['Modelo']}")
            lines.append(f"  Demanda semanal:     {row['Demanda Semanal']:.0f} unidades")
            lines.append(f"  Stock seguridad:     {row['Stock Seguridad']:.0f} unidades")
            lines.append(f"  Punto de reorden:    {row['Punto Reorden']:.0f} unidades")
            lines.append(f"  Cantidad a pedir:    {row['Cantidad Pedido']:.0f} unidades")
            lines.append(f"  Inventario promedio: {row['Inventario Promedio']:.0f} unidades")
            lines.append("")
        
        # Totales
        total_safety = standard_plans['Stock Seguridad'].sum()
        total_avg_inv = standard_plans['Inventario Promedio'].sum()
        
        lines.append("="*70)
        lines.append("TOTALES (Nivel 95%)")
        lines.append("="*70)
        lines.append(f"  Stock seguridad total:     {total_safety:.0f} unidades")
        lines.append(f"  Inventario promedio total: {total_avg_inv:.0f} unidades")
        lines.append("")
        lines.append("="*70)
        
        return "\n".join(lines)


if __name__ == '__main__':
    print("Módulo stock_calculator.py cargado")
