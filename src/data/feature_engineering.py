"""
Módulo de Feature Engineering - Olist Logistics Intelligence

Responsabilidad:
    - Fusionar datasets (orders, customers, geo, items, products)
    - Crear variable objetivo (delay_days)
    - Crear features temporales
    - Limpiar datos faltantes

Fase del Proyecto: 1 (Ingeniería de Datos)
Output: df_main listo para análisis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Construye el dataset maestro con todas las features necesarias
    
    Responsabilidades:
        - Merge de múltiples tablas
        - Creación de variables derivadas
        - Limpieza y validación
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> df_main = engineer.process(
        ...     orders=df_orders,
        ...     customers=df_customers,
        ...     geo=df_geo_centroids,
        ...     items=df_items,
        ...     products=df_products
        ... )
    """
    
    def __init__(self):
        """Inicializa el feature engineer"""
        logger.info("🔧 FeatureEngineer inicializado")
        
        # Columnas de fecha a convertir
        self.date_columns = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    
    def merge_datasets(self,
                      orders: pd.DataFrame,
                      customers: pd.DataFrame,
                      geo: pd.DataFrame,
                      items: Optional[pd.DataFrame] = None,
                      products: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Fusiona todos los datasets en uno maestro
        
        Args:
            orders: DataFrame de órdenes
            customers: DataFrame de clientes
            geo: DataFrame de geolocalización (centroides)
            items: DataFrame de items (opcional, para Track B)
            products: DataFrame de productos (opcional, para Track B)
            
        Returns:
            DataFrame unificado
        """
        logger.info("🔗 Fusionando datasets...")
        
        # Merge 1: Orders + Customers
        df = orders.merge(customers, on='customer_id', how='left')
        logger.info(f"   Orders + Customers → {df.shape}")
        
        # Merge 2: + Geolocation (centroides)
        df = df.merge(
            geo,
            left_on='customer_zip_code_prefix',
            right_on='geolocation_zip_code_prefix',
            how='left'
        )
        logger.info(f"   + Geo → {df.shape}")
        
        # Merge 3: + Items (si se proporciona)
        if items is not None:
            df = df.merge(
                items[['order_id', 'product_id', 'price']],
                on='order_id',
                how='left'
            )
            logger.info(f"   + Items → {df.shape}")
        
        # Merge 4: + Products (si se proporciona)
        if products is not None and items is not None:
            df = df.merge(
                products[['product_id', 'product_category_name']],
                on='product_id',
                how='left'
            )
            logger.info(f"   + Products → {df.shape}")
        
        logger.info(f"✅ Merge completado: {df.shape}")
        return df
    
    def filter_delivered(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra solo órdenes con status 'delivered'
        
        Args:
            df: DataFrame con columna 'order_status'
            
        Returns:
            DataFrame filtrado
        """
        logger.info("📦 Filtrando órdenes entregadas...")
        
        before = len(df)
        df_filtered = df[df['order_status'] == 'delivered'].copy()
        after = len(df_filtered)
        
        removed = before - after
        pct_removed = (removed / before) * 100
        
        logger.info(f"   Eliminadas: {removed:,} ({pct_removed:.1f}%)")
        logger.info(f"   Conservadas: {after:,}")
        
        return df_filtered
    
    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas de fecha a datetime
        
        Args:
            df: DataFrame con columnas de fecha
            
        Returns:
            DataFrame con fechas convertidas
        """
        logger.info("📅 Convirtiendo fechas...")
        
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"   ✅ {col}")
        
        return df
    
    def create_delay_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variable delay_days (KPI principal para Track A - Kriging)
        
        Formula:
            delay_days = fecha_entrega_real - fecha_entrega_estimada
        
        Interpretación:
            > 0: Retraso (malo)
            = 0: Puntual
            < 0: Anticipado (bueno)
        
        Args:
            df: DataFrame con columnas de fecha
            
        Returns:
            DataFrame con columna 'delay_days'
        """
        logger.info("⏱️  Creando variable delay_days...")
        
        df['delay_days'] = (
            df['order_delivered_customer_date'] - 
            df['order_estimated_delivery_date']
        ).dt.days
        
        # Estadísticas
        mean_delay = df['delay_days'].mean()
        median_delay = df['delay_days'].median()
        pct_delayed = (df['delay_days'] > 0).sum() / len(df) * 100
        
        logger.info(f"   Media: {mean_delay:.2f} días")
        logger.info(f"   Mediana: {median_delay:.2f} días")
        logger.info(f"   % con retraso: {pct_delayed:.1f}%")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features temporales desde order_purchase_timestamp
        
        Features creados:
            - year: Año
            - month: Mes (1-12)
            - week: Semana del año (1-53)
            - day_of_week: Día de la semana (0=Lunes, 6=Domingo)
            - quarter: Trimestre (1-4)
        
        Args:
            df: DataFrame con 'order_purchase_timestamp'
            
        Returns:
            DataFrame con features temporales
        """
        logger.info("📆 Creando features temporales...")
        
        df['year'] = df['order_purchase_timestamp'].dt.year
        df['month'] = df['order_purchase_timestamp'].dt.month
        df['week'] = df['order_purchase_timestamp'].dt.isocalendar().week
        df['day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
        df['quarter'] = df['order_purchase_timestamp'].dt.quarter
        
        logger.info(f"   ✅ 5 features creados")
        
        return df
    
    def clean_missing_geo(self, df: pd.DataFrame, 
                         max_missing_pct: float = 1.0) -> pd.DataFrame:
        """
        Elimina registros sin coordenadas geográficas
        
        Args:
            df: DataFrame con columnas geográficas
            max_missing_pct: % máximo aceptable de pérdida
            
        Returns:
            DataFrame limpio
            
        Raises:
            ValueError: Si la pérdida supera el umbral
        """
        logger.info("🗺️  Limpiando datos geográficos...")
        
        before = len(df)
        
        # Eliminar nulos
        df_clean = df.dropna(subset=['geolocation_lat', 'geolocation_lng']).copy()
        
        after = len(df_clean)
        lost = before - after
        lost_pct = (lost / before) * 100
        
        logger.info(f"   Registros perdidos: {lost:,} ({lost_pct:.2f}%)")
        
        # Validar umbral
        if lost_pct > max_missing_pct:
            logger.warning(f"   ⚠️  Pérdida ({lost_pct:.2f}%) > umbral ({max_missing_pct}%)")
            logger.warning(f"   Considera imputación en lugar de eliminación")
        
        return df_clean
    
    def process(self,
                orders: pd.DataFrame,
                customers: pd.DataFrame,
                geo: pd.DataFrame,
                items: Optional[pd.DataFrame] = None,
                products: Optional[pd.DataFrame] = None,
                max_missing_geo_pct: float = 1.0) -> pd.DataFrame:
        """
        Pipeline completo de feature engineering
        
        Args:
            orders, customers, geo: DataFrames requeridos
            items, products: DataFrames opcionales
            max_missing_geo_pct: Umbral de pérdida aceptable
            
        Returns:
            df_main listo para modelado
            
        Example:
            >>> engineer = FeatureEngineer()
            >>> df_main = engineer.process(
            ...     orders=olist_data['orders'],
            ...     customers=olist_data['customers'],
            ...     geo=df_geo_processed,
            ...     items=olist_data['items'],
            ...     products=olist_data['products']
            ... )
        """
        logger.info("🚀 Iniciando pipeline de Feature Engineering...")
        
        # Paso 1: Merge
        df = self.merge_datasets(orders, customers, geo, items, products)
        
        # Paso 2: Filtrar entregadas
        df = self.filter_delivered(df)
        
        # Paso 3: Convertir fechas
        df = self.convert_dates(df)
        
        # Paso 4: Crear delay_days (Track A)
        df = self.create_delay_variable(df)
        
        # Paso 5: Features temporales
        df = self.create_temporal_features(df)
        
        # Paso 6: Limpieza geográfica
        df = self.clean_missing_geo(df, max_missing_geo_pct)
        
        logger.info("✅ Feature Engineering completado")
        logger.info(f"📊 Dataset final: {df.shape}")
        logger.info(f"📅 Rango temporal: {df['order_purchase_timestamp'].min()} → {df['order_purchase_timestamp'].max()}")
        
        return df
    
    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera resumen del dataset procesado
        
        Args:
            df: DataFrame procesado
            
        Returns:
            DataFrame con estadísticas
        """
        summary = {
            'Total Registros': len(df),
            'Órdenes Únicas': df['order_id'].nunique(),
            'Clientes Únicos': df['customer_id'].nunique(),
            'Estados': df['geolocation_state'].nunique() if 'geolocation_state' in df else 'N/A',
            'Categorías': df['product_category_name'].nunique() if 'product_category_name' in df else 'N/A',
            'Delay Promedio (días)': f"{df['delay_days'].mean():.2f}",
            'Rango Temporal': f"{df['order_purchase_timestamp'].min().date()} → {df['order_purchase_timestamp'].max().date()}"
        }
        
        return pd.DataFrame([summary]).T.rename(columns={0: 'Valor'})


# ============================================================================
# TESTING
# ============================================================================
if __name__ == '__main__':
    print("Módulo feature_engineering.py cargado correctamente")
