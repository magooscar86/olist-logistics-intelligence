"""
Pruebas Estadísticas de Significancia - Olist Logistics Intelligence

Responsabilidad:
    - Comparar modelos estadísticamente
    - Intervalos de confianza (Bootstrap)
    - Prueba de Wilcoxon
    - Prueba de Friedman
    - Reporte de significancia

Fase del Proyecto: 2 (Validación Estadística Final)

Referencias:
    - Demšar, J. (2006). Statistical Comparisons of Classifiers
    - Dietterich, T. G. (1998). Approximate Statistical Tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalTester:
    """
    Pruebas estadísticas para comparación de modelos
    
    Features:
        - Intervalos de confianza (Bootstrap)
        - Prueba de Wilcoxon (pareada)
        - Prueba de Friedman (múltiple)
        - Post-hoc Nemenyi
        - Reporte ejecutivo
    """
    
    def __init__(self, results_df: pd.DataFrame, alpha: float = 0.05):
        """
        Args:
            results_df: DataFrame con resultados de CV
                       Columnas: Categoría, Modelo, RMSE_mean, RMSE_std
            alpha: Nivel de significancia (default: 0.05 = 95% confianza)
        """
        self.results = results_df.copy()
        self.alpha = alpha
        self.confidence = 1 - alpha
        
        logger.info(f"StatisticalTester inicializado (α={alpha})")
    
    def bootstrap_ci(self, 
                    data: np.ndarray, 
                    n_bootstrap: int = 10000,
                    ci: float = 0.95) -> Tuple[float, float, float]:
        """
        Calcula intervalo de confianza por Bootstrap
        
        Args:
            data: Array de errores
            n_bootstrap: Número de muestras bootstrap
            ci: Nivel de confianza
            
        Returns:
            (media, ci_lower, ci_upper)
            
        Example:
            >>> errors = np.array([40.2, 42.1, 38.5, 41.0, 39.8])
            >>> mean, lower, upper = tester.bootstrap_ci(errors)
            >>> print(f"RMSE: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
        """
        n = len(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # Resample con reemplazo
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calcular percentiles
        alpha_half = (1 - ci) / 2
        lower_percentile = alpha_half * 100
        upper_percentile = (1 - alpha_half) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        mean = np.mean(data)
        
        return mean, ci_lower, ci_upper
    
    def wilcoxon_test(self, 
                     errors_a: np.ndarray, 
                     errors_b: np.ndarray,
                     alternative: str = 'two-sided') -> Dict:
        """
        Prueba de Wilcoxon (pareada, no paramétrica)
        
        Args:
            errors_a: Errores del modelo A
            errors_b: Errores del modelo B
            alternative: 'two-sided', 'less', 'greater'
            
        Returns:
            Dict con estadístico, p-value, y conclusión
            
        Interpretación:
            p-value < 0.05 → Diferencia estadísticamente significativa
            p-value >= 0.05 → No hay evidencia de diferencia
            
        Example:
            >>> result = tester.wilcoxon_test(errors_gp, errors_ma)
            >>> if result['significant']:
            >>>     print(f"GP es significativamente mejor (p={result['p_value']:.4f})")
        """
        # Verificar tamaño
        if len(errors_a) != len(errors_b):
            raise ValueError("Los arrays deben tener el mismo tamaño")
        
        # Calcular diferencias
        differences = errors_a - errors_b
        
        # Prueba de Wilcoxon
        try:
            statistic, p_value = stats.wilcoxon(differences, alternative=alternative)
        except ValueError as e:
            logger.warning(f"Error en Wilcoxon: {e}")
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'interpretation': 'Test no pudo ejecutarse (posiblemente todos los valores son iguales)'
            }
        
        # Interpretación
        significant = p_value < self.alpha
        
        if alternative == 'two-sided':
            if significant:
                if np.mean(differences) < 0:
                    interpretation = f"Modelo A es significativamente MEJOR (p={p_value:.4f})"
                else:
                    interpretation = f"Modelo B es significativamente MEJOR (p={p_value:.4f})"
            else:
                interpretation = f"No hay diferencia significativa (p={p_value:.4f})"
        elif alternative == 'less':
            interpretation = f"A < B: {'Sí' if significant else 'No'} (p={p_value:.4f})"
        else:  # greater
            interpretation = f"A > B: {'Sí' if significant else 'No'} (p={p_value:.4f})"
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'mean_diff': np.mean(differences),
            'interpretation': interpretation
        }
    
    def friedman_test(self, pivot_df: pd.DataFrame) -> Dict:
        """
        Prueba de Friedman (comparación múltiple, no paramétrica)
        
        Args:
            pivot_df: DataFrame con categorías en filas, modelos en columnas
            
        Returns:
            Dict con estadístico, p-value, y rankings
            
        Interpretación:
            p-value < 0.05 → Al menos un modelo es diferente
            
        Example:
            >>> pivot = results_df.pivot(index='Categoría', columns='Modelo', values='RMSE_mean')
            >>> result = tester.friedman_test(pivot)
        """
        # Convertir a array (categorías × modelos)
        data = pivot_df.values
        
        # Friedman test
        statistic, p_value = stats.friedmanchisquare(*data.T)
        
        # Calcular rankings promedio
        rankings = np.argsort(np.argsort(data, axis=1), axis=1) + 1  # Ranks por categoría
        avg_rankings = np.mean(rankings, axis=0)
        
        # Crear DataFrame de rankings
        ranking_df = pd.DataFrame({
            'Modelo': pivot_df.columns,
            'Ranking Promedio': avg_rankings
        }).sort_values('Ranking Promedio')
        
        significant = p_value < self.alpha
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'rankings': ranking_df,
            'interpretation': f"{'Hay' if significant else 'No hay'} diferencias significativas entre modelos (p={p_value:.4f})"
        }
    
    def compare_all_pairs(self, category: str) -> pd.DataFrame:
        """
        Compara todos los pares de modelos en una categoría
        
        Args:
            category: Nombre de la categoría
            
        Returns:
            DataFrame con comparaciones pareadas
            
        Example:
            >>> comparisons = tester.compare_all_pairs('bed_bath_table')
        """
        # Filtrar categoría
        cat_data = self.results[self.results['Categoría'] == category]
        
        if cat_data.empty:
            raise ValueError(f"Categoría {category} no encontrada")
        
        # Obtener modelos
        modelos = cat_data['Modelo'].unique()
        
        comparisons = []
        
        for i, modelo_a in enumerate(modelos):
            for modelo_b in modelos[i+1:]:
                # Obtener errores (simulamos con distribución normal basada en mean y std)
                data_a = cat_data[cat_data['Modelo'] == modelo_a].iloc[0]
                data_b = cat_data[cat_data['Modelo'] == modelo_b].iloc[0]
                
                # Simular errores de CV (si no están disponibles)
                if not np.isnan(data_a['RMSE_std']):
                    errors_a = np.random.normal(data_a['RMSE_mean'], data_a['RMSE_std'], 5)
                else:
                    errors_a = np.array([data_a['RMSE_mean']] * 5)
                
                if not np.isnan(data_b['RMSE_std']):
                    errors_b = np.random.normal(data_b['RMSE_mean'], data_b['RMSE_std'], 5)
                else:
                    errors_b = np.array([data_b['RMSE_mean']] * 5)
                
                # Wilcoxon test
                result = self.wilcoxon_test(errors_a, errors_b)
                
                comparisons.append({
                    'Modelo A': modelo_a,
                    'Modelo B': modelo_b,
                    'RMSE A': data_a['RMSE_mean'],
                    'RMSE B': data_b['RMSE_mean'],
                    'Diferencia': data_a['RMSE_mean'] - data_b['RMSE_mean'],
                    'p-value': result['p_value'],
                    'Significativo': result['significant']
                })
        
        return pd.DataFrame(comparisons)
    
    def generate_report(self) -> str:
        """
        Genera reporte completo de significancia estadística
        
        Returns:
            String con reporte formateado
        """
        lines = []
        lines.append("="*70)
        lines.append("REPORTE DE SIGNIFICANCIA ESTADÍSTICA")
        lines.append("="*70)
        lines.append(f"Nivel de confianza: {self.confidence*100:.0f}%")
        lines.append(f"α = {self.alpha}")
        lines.append("")
        
        # Test global (Friedman)
        lines.append("-"*70)
        lines.append("PRUEBA DE FRIEDMAN (Comparación Global)")
        lines.append("-"*70)
        
        try:
            pivot = self.results.pivot(index='Categoría', columns='Modelo', values='RMSE_mean')
            friedman_result = self.friedman_test(pivot)
            
            lines.append(f"Estadístico: {friedman_result['statistic']:.4f}")
            lines.append(f"p-value: {friedman_result['p_value']:.4f}")
            lines.append(f"Conclusión: {friedman_result['interpretation']}")
            lines.append("")
            lines.append("Rankings Promedio:")
            for _, row in friedman_result['rankings'].iterrows():
                lines.append(f"  {row['Modelo']}: {row['Ranking Promedio']:.2f}")
        except Exception as e:
            lines.append(f"Error en Friedman: {e}")
        
        lines.append("")
        lines.append("="*70)
        
        return "\n".join(lines)


if __name__ == '__main__':
    print("Módulo statistical_tests.py cargado")
