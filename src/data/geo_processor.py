"""
Módulo de Procesamiento Geoespacial - Olist Logistics Intelligence

Responsabilidad:
    - Filtrar coordenadas outliers (bounding box)
    - Calcular centroides por código postal
    - Reducir dimensionalidad del dataset geo

Fase del Proyecto: 1 (Ingeniería de Datos)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoProcessor:
    """
    Procesa datos geoespaciales del dataset Olist
    
    Responsabilidades:
        - Filtrado de outliers geográficos
        - Cálculo de centroides por zona postal
        - Validación de coordenadas
    
    Example:
        >>> from config import BRAZIL_BOUNDS
        >>> processor = GeoProcessor(bounds=BRAZIL_BOUNDS)
        >>> df_clean = processor.filter_outliers(df_geo)
        >>> df_centroids = processor.calculate_centroids(df_clean)
    """
    
    def __init__(self, bounds: Dict[str, float]):
        """
        Inicializa el procesador geoespacial
        
        Args:
            bounds: Diccionario con límites geográficos
                   {'lat_min': -34, 'lat_max': 6, 'lng_min': -74, 'lng_max': -34}
        """
        self.bounds = bounds
        logger.info(f"🌍 GeoProcessor inicializado con bounds: {bounds}")
    
    def filter_outliers(self, df_geo: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra coordenadas fuera del bounding box
        
        Args:
            df_geo: DataFrame con columnas 'geolocation_lat' y 'geolocation_lng'
            
        Returns:
            DataFrame filtrado sin outliers
            
        Example:
            >>> df_clean = processor.filter_outliers(df_geo)
            >>> # Elimina coordenadas en océano, otros países, etc.
        """
        logger.info("🧹 Filtrando outliers geográficos...")
        
        original_size = len(df_geo)
        
        # Aplicar bounding box
        df_clean = df_geo[
            (df_geo['geolocation_lat'].between(self.bounds['lat_min'], self.bounds['lat_max'])) &
            (df_geo['geolocation_lng'].between(self.bounds['lng_min'], self.bounds['lng_max']))
        ].copy()
        
        outliers_removed = original_size - len(df_clean)
        pct_removed = (outliers_removed / original_size) * 100
        
        logger.info(f"   Outliers eliminados: {outliers_removed:,} ({pct_removed:.2f}%)")
        logger.info(f"   Registros válidos: {len(df_clean):,}")
        
        return df_clean
    
    def calculate_centroids(self, df_geo: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula centroides por código postal
        
        Estrategia:
            - Agrupa por 'geolocation_zip_code_prefix'
            - Calcula media de lat/lng (centroide geométrico)
            - Toma primer valor de ciudad/estado
        
        Args:
            df_geo: DataFrame limpio con coordenadas
            
        Returns:
            DataFrame con centroides únicos por zip code
            
        Reducción esperada:
            ~1,000,000 filas → ~19,000 centroides
        """
        logger.info("📍 Calculando centroides por código postal...")
        
        original_size = len(df_geo)
        
        # Agrupar y calcular centroides
        df_centroids = df_geo.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',      # Centroide: promedio de latitudes
            'geolocation_lng': 'mean',      # Centroide: promedio de longitudes
            'geolocation_city': 'first',    # Primera ciudad del grupo
            'geolocation_state': 'first'    # Primer estado del grupo
        }).reset_index()
        
        reduction = len(df_centroids)
        reduction_pct = (1 - reduction / original_size) * 100
        
        logger.info(f"   Reducción dimensional: {original_size:,} → {reduction:,}")
        logger.info(f"   Compresión: {reduction_pct:.1f}%")
        
        return df_centroids
    
    def process(self, df_geo: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo: filtrar + calcular centroides
        
        Args:
            df_geo: DataFrame crudo de geolocalización
            
        Returns:
            DataFrame procesado con centroides únicos
            
        Example:
            >>> processor = GeoProcessor(bounds=BRAZIL_BOUNDS)
            >>> df_geo_processed = processor.process(df_geo_raw)
        """
        logger.info("🚀 Iniciando pipeline geoespacial completo...")
        
        # Paso 1: Filtrar outliers
        df_clean = self.filter_outliers(df_geo)
        
        # Paso 2: Calcular centroides
        df_centroids = self.calculate_centroids(df_clean)
        
        logger.info("✅ Pipeline geoespacial completado")
        
        return df_centroids
    
    def validate_coordinates(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Valida la calidad de las coordenadas
        
        Args:
            df: DataFrame con columnas de coordenadas
            
        Returns:
            Diccionario con estadísticas de validación
        """
        stats = {
            'total_records': len(df),
            'missing_lat': df['geolocation_lat'].isna().sum(),
            'missing_lng': df['geolocation_lng'].isna().sum(),
            'lat_range': (df['geolocation_lat'].min(), df['geolocation_lat'].max()),
            'lng_range': (df['geolocation_lng'].min(), df['geolocation_lng'].max()),
        }
        
        stats['missing_pct'] = (
            (stats['missing_lat'] + stats['missing_lng']) / 
            (2 * stats['total_records'])
        ) * 100
        
        return stats


# ============================================================================
# TESTING
# ============================================================================
if __name__ == '__main__':
    from config import BRAZIL_BOUNDS
    
    # Simular datos
    df_test = pd.DataFrame({
        'geolocation_zip_code_prefix': [1000, 1000, 2000],
        'geolocation_lat': [-23.5, -23.6, 100],  # 100 es outlier
        'geolocation_lng': [-46.6, -46.5, -46.0],
        'geolocation_city': ['São Paulo', 'São Paulo', 'Test'],
        'geolocation_state': ['SP', 'SP', 'XX']
    })
    
    processor = GeoProcessor(bounds=BRAZIL_BOUNDS)
    df_processed = processor.process(df_test)
    
    print("\n📊 Resultado:")
    print(df_processed)
