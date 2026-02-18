"""
Módulo de Procesamiento de Series Temporales - Olist Logistics Intelligence

Responsabilidad:
    - Agregación temporal (semanal)
    - Filtrado de período denso
    - Preparación de X, y para modelos
    - Soporte para análisis total y por categoría

Fase del Proyecto: 2 (Forecasting)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """
    Procesa datos transaccionales en series temporales para forecasting
    
    Capacidades:
        - Agregación semanal
        - Filtrado de períodos densos
        - Preparación de arrays X, y para GP
        - Análisis total o por categoría
    
    Example:
        >>> from config import TIME_SERIES_CONFIG
        >>> processor = TimeSeriesProcessor(
        ...     start_date=TIME_SERIES_CONFIG['start_date'],
        ...     end_date=TIME_SERIES_CONFIG['end_date']
        ... )
        >>> X, y, dates = processor.process(df_main)
    """
    
    def __init__(self, 
                 start_date: str = '2017-01-01',
                 end_date: str = '2018-08-31',
                 frequency: str = 'W',
                 min_weeks: int = 10):
        """
        Inicializa el procesador de series temporales
        
        Args:
            start_date: Inicio del período a analizar
            end_date: Fin del período a analizar
            frequency: Frecuencia de agregación ('W' = semanal)
            min_weeks: Mínimo de semanas requeridas para validar serie
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
        self.min_weeks = min_weeks
        
        logger.info("📅 TimeSeriesProcessor inicializado")
        logger.info(f"   Período: {start_date} → {end_date}")
        logger.info(f"   Frecuencia: {frequency}")
    
    def aggregate_weekly(self, 
                        df: pd.DataFrame,
                        date_column: str = 'order_purchase_timestamp',
                        category: Optional[str] = None,
                        category_column: str = 'product_category_name') -> pd.Series:
        """
        Agrega datos semanalmente
        
        Args:
            df: DataFrame con datos transaccionales
            date_column: Columna con fechas
            category: (Opcional) Nombre de categoría para filtrar
            category_column: Nombre de la columna de categorías
            
        Returns:
            Serie temporal con conteo semanal de órdenes
            
        Example:
            >>> # Demanda total
            >>> y_total = processor.aggregate_weekly(df_main)
            >>> 
            >>> # Demanda por categoría
            >>> y_bed = processor.aggregate_weekly(df_main, category='bed_bath_table')
        """
        # Filtrar por categoría si se especifica
        if category is not None:
            if category_column not in df.columns:
                raise ValueError(f"Columna '{category_column}' no existe")
            
            df_filtered = df[df[category_column] == category].copy()
            logger.info(f"   Filtrando categoría: {category}")
            logger.info(f"   Registros: {len(df):,} → {len(df_filtered):,}")
        else:
            df_filtered = df.copy()
            logger.info("   Agregación total (sin filtro)")
        
        # Validar columna de fecha
        if date_column not in df_filtered.columns:
            raise ValueError(f"Columna '{date_column}' no existe")
        
        # Asegurar que es datetime
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[date_column]):
            df_filtered[date_column] = pd.to_datetime(df_filtered[date_column])
        
        # Agregar semanalmente
        df_ts = df_filtered.set_index(date_column)
        y = df_ts.resample(self.frequency)['order_id'].count()
        
        logger.info(f"   Serie temporal creada: {len(y)} períodos")
        
        return y
    
    def filter_dense_period(self, y: pd.Series) -> pd.Series:
        """
        Filtra el período denso (2017-2018)
        
        Elimina:
            - Semanas iniciales con pocos datos (ramp-up)
            - Semanas finales incompletas (tail-off)
        
        Args:
            y: Serie temporal completa
            
        Returns:
            Serie temporal filtrada
        """
        logger.info("   Aplicando filtro de período denso...")
        logger.info(f"   Antes: {len(y)} semanas")
        
        mask = (y.index >= self.start_date) & (y.index <= self.end_date)
        y_filtered = y[mask]
        
        logger.info(f"   Después: {len(y_filtered)} semanas")
        
        return y_filtered
    
    def prepare_xy(self, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepara arrays X, y para entrenamiento
        
        Args:
            y: Serie temporal
            
        Returns:
            X: Índices temporales (0, 1, 2, ..., N)
            y_values: Valores de demanda
            dates: Índice de fechas original
            
        Example:
            >>> X, y_values, dates = processor.prepare_xy(y_series)
            >>> gp_model.fit(X, y_values)
        """
        X = np.arange(len(y)).reshape(-1, 1)
        y_values = y.values
        dates = y.index
        
        logger.info(f"   Arrays preparados: X={X.shape}, y={y_values.shape}")
        
        return X, y_values, dates
    
    def validate_series(self, y: pd.Series, category: Optional[str] = None) -> bool:
        """
        Valida que la serie tenga suficientes datos
        
        Args:
            y: Serie temporal
            category: Nombre de categoría (para logging)
            
        Returns:
            True si la serie es válida
        """
        n_weeks = len(y)
        
        if n_weeks < self.min_weeks:
            cat_name = category if category else "Total"
            logger.warning(f"   ⚠️  {cat_name}: Solo {n_weeks} semanas (mínimo: {self.min_weeks})")
            return False
        
        return True
    
    def process(self,
                df: pd.DataFrame,
                category: Optional[str] = None,
                validate: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Pipeline completo: agregación → filtrado → preparación
        
        Args:
            df: DataFrame con datos transaccionales
            category: (Opcional) Categoría para analizar
            validate: Si True, valida cantidad mínima de datos
            
        Returns:
            X, y, dates listos para modelado
            
        Raises:
            ValueError: Si la serie no cumple validación
            
        Example:
            >>> # Demanda total
            >>> X, y, dates = processor.process(df_main)
            >>> 
            >>> # Demanda por categoría
            >>> X, y, dates = processor.process(df_main, category='bed_bath_table')
        """
        cat_label = f"'{category}'" if category else "Total"
        logger.info(f"🔄 Procesando serie: {cat_label}")
        
        # Paso 1: Agregar
        y_series = self.aggregate_weekly(df, category=category)
        
        # Paso 2: Filtrar período denso
        y_series = self.filter_dense_period(y_series)
        
        # Paso 3: Validar
        if validate and not self.validate_series(y_series, category):
            raise ValueError(f"Serie '{cat_label}' no cumple mínimo de {self.min_weeks} semanas")
        
        # Paso 4: Preparar X, y
        X, y, dates = self.prepare_xy(y_series)
        
        logger.info(f"✅ Serie procesada: {len(y)} semanas")
        
        return X, y, dates
    
    def process_multiple_categories(self,
                                   df: pd.DataFrame,
                                   categories: list,
                                   skip_invalid: bool = True) -> Dict[str, Tuple]:
        """
        Procesa múltiples categorías en batch
        
        Args:
            df: DataFrame con datos
            categories: Lista de categorías a procesar
            skip_invalid: Si True, omite categorías con datos insuficientes
            
        Returns:
            Diccionario {categoria: (X, y, dates)}
            
        Example:
            >>> top_cats = ['bed_bath_table', 'health_beauty', 'sports_leisure']
            >>> results = processor.process_multiple_categories(df_main, top_cats)
        """
        logger.info(f"🔄 Procesando {len(categories)} categorías...")
        
        results = {}
        
        for i, cat in enumerate(categories, 1):
            logger.info(f"\n[{i}/{len(categories)}] {cat}")
            
            try:
                X, y, dates = self.process(df, category=cat, validate=True)
                results[cat] = (X, y, dates)
                
            except ValueError as e:
                if skip_invalid:
                    logger.warning(f"   ⚠️  Omitiendo: {e}")
                else:
                    raise
        
        logger.info(f"\n✅ Procesadas exitosamente: {len(results)}/{len(categories)}")
        
        return results


# ============================================================================
# TESTING
# ============================================================================
if __name__ == '__main__':
    print("Módulo time_series_processor.py cargado correctamente")
