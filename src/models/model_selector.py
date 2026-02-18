"""
Sistema de Selección Automática de Modelos - Olist Logistics Intelligence

Responsabilidad:
    - Analizar resultados del torneo/CV
    - Seleccionar mejor modelo por categoría
    - Generar recomendaciones de negocio
    - Extraer insights accionables

Fase del Proyecto: 2 (Validación Final)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoModelSelector:
    """
    Selector inteligente de modelos basado en resultados de validación
    
    Criterios de selección:
        1. RMSE mínimo (precisión)
        2. Estabilidad (baja desviación estándar)
        3. Mejora vs baseline
        4. Complejidad justificada
    """
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Args:
            results_df: DataFrame con resultados de validación
                       Columnas: Categoría, Modelo, Método, RMSE_mean, RMSE_std
        """
        self.results = results_df.copy()
        self.selections = {}
        self.summary = None
        
        logger.info(f"AutoModelSelector inicializado con {len(results_df)} resultados")
    
    def analyze_category(self, category: str) -> Dict:
        """
        Analiza una categoría y selecciona el mejor modelo
        
        Args:
            category: Nombre de la categoría
            
        Returns:
            Dict con modelo seleccionado, métricas e insights
        """
        # Filtrar resultados de esta categoría
        cat_results = self.results[self.results['Categoría'] == category].copy()
        
        if cat_results.empty:
            raise ValueError(f"Categoría {category} no encontrada")
        
        # Ordenar por RMSE
        cat_results = cat_results.sort_values('RMSE_mean')
        
        # Ganador
        winner = cat_results.iloc[0]
        
        # Baseline (Media Móvil)
        baseline = cat_results[cat_results['Modelo'] == 'Media Móvil']
        if baseline.empty:
            baseline_rmse = np.nan
            mejora_pct = 0
        else:
            baseline_rmse = baseline.iloc[0]['RMSE_mean']
            mejora_pct = ((baseline_rmse - winner['RMSE_mean']) / baseline_rmse) * 100
        
        # Clasificar volatilidad
        if not np.isnan(winner['RMSE_std']):
            cv_coef = (winner['RMSE_std'] / winner['RMSE_mean']) * 100
            if cv_coef < 10:
                estabilidad = 'MUY ESTABLE'
            elif cv_coef < 20:
                estabilidad = 'ESTABLE'
            elif cv_coef < 30:
                estabilidad = 'MODERADA'
            else:
                estabilidad = 'INESTABLE'
        else:
            cv_coef = np.nan
            estabilidad = 'N/A (Hold-out)'
        
        # Horizonte de planificación basado en RMSE
        if winner['RMSE_mean'] < 30:
            horizonte = 'LARGO PLAZO (3+ meses)'
            nivel_confianza = 'ALTO'
        elif winner['RMSE_mean'] < 45:
            horizonte = 'MEDIANO PLAZO (1-3 meses)'
            nivel_confianza = 'MEDIO'
        else:
            horizonte = 'CORTO PLAZO (<1 mes)'
            nivel_confianza = 'BAJO'
        
        # Justificación técnica
        justificacion = self._generate_justification(winner, baseline_rmse, mejora_pct, category)
        
        # Construir recomendación
        recomendacion = {
            'categoria': category,
            'modelo_seleccionado': winner['Modelo'],
            'metodo': winner['Método'],
            'rmse': winner['RMSE_mean'],
            'rmse_std': winner['RMSE_std'],
            'estabilidad': estabilidad,
            'cv_coef': cv_coef,
            'mejora_vs_baseline': mejora_pct,
            'horizonte_planificacion': horizonte,
            'nivel_confianza': nivel_confianza,
            'justificacion': justificacion
        }
        
        self.selections[category] = recomendacion
        
        return recomendacion
    
    def _generate_justification(self, winner: pd.Series, baseline_rmse: float, 
                                mejora: float, category: str) -> str:
        """Genera justificación técnica de la selección"""
        modelo = winner['Modelo']
        rmse = winner['RMSE_mean']
        
        # Baseline ganó
        if modelo == 'Media Móvil':
            return (f"Categoría extremadamente volátil sin estructura predecible. "
                   f"Modelos complejos sobreajustan. Parsimonia (Occam's Razor) "
                   f"recomienda baseline simple. RMSE={rmse:.2f}.")
        
        # GP ganó
        elif modelo == 'GP Puro':
            return (f"Categoría estable con tendencia y estacionalidad claras. "
                   f"GP captura patrones de largo plazo eficientemente. "
                   f"Mejora vs baseline: {mejora:.1f}%. RMSE={rmse:.2f}.")
        
        # LGBM ganó
        elif modelo == 'LightGBM Puro':
            return (f"Patrones complejos de corto plazo detectados. "
                   f"LGBM con features de lags y rolling captura dinámica temporal. "
                   f"Mejora vs baseline: {mejora:.1f}%. RMSE={rmse:.2f}.")
        
        # Holt ganó
        elif modelo == 'Holt-Winters':
            return (f"Tendencia lineal fuerte sin estacionalidad compleja. "
                   f"Holt-Winters óptimo para este patrón. "
                   f"Mejora vs baseline: {mejora:.1f}%. RMSE={rmse:.2f}.")
        
        # Híbrido (no debería pasar)
        elif 'Híbrido' in modelo:
            return (f"Combinación GP+LGBM seleccionada. "
                   f"Requiere validación adicional. RMSE={rmse:.2f}.")
        
        else:
            return f"Modelo {modelo} seleccionado. RMSE={rmse:.2f}."
    
    def analyze_all(self) -> pd.DataFrame:
        """
        Analiza todas las categorías y genera resumen
        
        Returns:
            DataFrame con selecciones y recomendaciones
        """
        categories = self.results['Categoría'].unique()
        
        logger.info(f"Analizando {len(categories)} categorías...")
        
        for cat in categories:
            self.analyze_category(cat)
        
        # Convertir a DataFrame
        self.summary = pd.DataFrame(self.selections).T.reset_index(drop=True)
        
        logger.info("Análisis completo")
        
        return self.summary
    
    def get_executive_summary(self) -> str:
        """
        Genera reporte ejecutivo en texto
        
        Returns:
            String con reporte formateado
        """
        if self.summary is None:
            return "No hay análisis. Ejecuta analyze_all() primero."
        
        lines = []
        lines.append("="*70)
        lines.append("REPORTE EJECUTIVO: SELECCIÓN AUTOMÁTICA DE MODELOS")
        lines.append("="*70)
        lines.append("")
        
        for _, row in self.summary.iterrows():
            lines.append(f"[{row['categoria'].upper()}]")
            lines.append(f"  Modelo:      {row['modelo_seleccionado']}")
            
            if not np.isnan(row['rmse_std']):
                lines.append(f"  RMSE:        {row['rmse']:.2f} ± {row['rmse_std']:.2f}")
            else:
                lines.append(f"  RMSE:        {row['rmse']:.2f}")
            
            lines.append(f"  Estabilidad: {row['estabilidad']}")
            lines.append(f"  Mejora:      {row['mejora_vs_baseline']:+.1f}%")
            lines.append(f"  Horizonte:   {row['horizonte_planificacion']}")
            lines.append(f"  Confianza:   {row['nivel_confianza']}")
            lines.append(f"  → {row['justificacion']}")
            lines.append("")
        
        # Estadísticas generales
        lines.append("="*70)
        lines.append("ESTADÍSTICAS GENERALES")
        lines.append("="*70)
        lines.append("")
        
        model_counts = self.summary['modelo_seleccionado'].value_counts()
        lines.append("Distribución de modelos:")
        for model, count in model_counts.items():
            pct = (count / len(self.summary)) * 100
            lines.append(f"  • {model}: {count} categorías ({pct:.1f}%)")
        
        lines.append("")
        avg_mejora = self.summary['mejora_vs_baseline'].mean()
        lines.append(f"Mejora promedio vs baseline: {avg_mejora:.1f}%")
        
        lines.append("")
        lines.append("="*70)
        
        return "\n".join(lines)
    
    def get_production_config(self) -> Dict:
        """
        Genera configuración para producción
        
        Returns:
            Dict {categoria: modelo_config}
        """
        if self.summary is None:
            raise ValueError("Ejecuta analyze_all() primero")
        
        config = {}
        
        for _, row in self.summary.iterrows():
            config[row['categoria']] = {
                'model_type': row['modelo_seleccionado'],
                'expected_rmse': row['rmse'],
                'confidence_level': row['nivel_confianza'],
                'retraining_frequency': self._get_retraining_frequency(row['horizonte_planificacion'])
            }
        
        return config
    
    def _get_retraining_frequency(self, horizonte: str) -> str:
        """Recomienda frecuencia de re-entrenamiento"""
        if 'LARGO' in horizonte:
            return 'Mensual'
        elif 'MEDIANO' in horizonte:
            return 'Quincenal'
        else:
            return 'Semanal'


if __name__ == '__main__':
    print("Módulo model_selector.py cargado")
