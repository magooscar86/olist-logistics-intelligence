"""
Módulo de Carga de Datos - Olist Logistics Intelligence

Responsabilidad:
    - Descomprimir archive.zip
    - Cargar todos los CSVs en DataFrames
    - Validar que existan los archivos necesarios

Versión: Google Drive
"""

import pandas as pd
import zipfile
import os
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OlistDataLoader:
    """
    Clase para gestionar la carga de datos del dataset Olist
    
    Attributes:
        raw_data_path (Path): Ruta donde están los CSVs
        zip_path (Path): Ruta del archivo ZIP (si existe)
    
    Example:
        >>> loader = OlistDataLoader(
        ...     raw_data_path='data/raw',
        ...     zip_path='data/raw/archive.zip'
        ... )
        >>> data = loader.load_all()
        >>> print(data['orders'].shape)
    """
    
    def __init__(self, raw_data_path: str, zip_path: Optional[str] = None):
        """
        Inicializa el cargador de datos
        
        Args:
            raw_data_path: Ruta donde se extraerán/están los CSVs
            zip_path: Ruta del archivo ZIP (opcional)
        """
        self.raw_data_path = Path(raw_data_path)
        self.zip_path = Path(zip_path) if zip_path else None
        
        self.required_files = [
            'olist_orders_dataset',
            'olist_customers_dataset',
            'olist_geolocation_dataset',
            'olist_order_items_dataset',
            'olist_products_dataset',
            'olist_sellers_dataset'
        ]
    
    def extract_zip(self) -> bool:
        """
        Extrae el archivo ZIP si existe
        
        Returns:
            True si se extrajo correctamente
        """
        if self.zip_path is None:
            logger.info("No se especificó archivo ZIP")
            return False
        
        if not self.zip_path.exists():
            raise FileNotFoundError(f"❌ No se encontró: {self.zip_path}")
        
        logger.info(f"📂 Descomprimiendo {self.zip_path.name}...")
        
        try:
            self.raw_data_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_path)
            
            logger.info("✅ Descompresión exitosa")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al descomprimir: {e}")
            raise
    
    def verify_files(self) -> bool:
        """
        Verifica que todos los archivos necesarios existan
        
        Returns:
            True si todos los archivos existen
        """
        logger.info("🔍 Verificando archivos...")
        
        missing_files = []
        
        for file_base in self.required_files:
            file_path = self.raw_data_path / f"{file_base}.csv"
            
            if not file_path.exists():
                missing_files.append(file_base)
                logger.warning(f"   ⚠️  Falta: {file_base}.csv")
            else:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"   ✅ {file_base}.csv ({size_mb:.2f} MB)")
        
        if missing_files:
            raise FileNotFoundError(
                f"❌ Faltan archivos: {missing_files}"
            )
        
        logger.info(f"✅ Todos los archivos verificados ({len(self.required_files)} archivos)")
        return True
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Carga un CSV específico
        
        Args:
            filename: Nombre del archivo (sin extensión)
            
        Returns:
            DataFrame con los datos
        """
        filepath = self.raw_data_path / f"{filename}.csv"
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"   Cargado: {filename} → {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Error cargando {filename}: {e}")
            raise
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Carga todos los CSVs necesarios
        
        Returns:
            Diccionario con {nombre: DataFrame}
        """
        logger.info("⏳ Cargando todos los datasets...")
        
        self.verify_files()
        
        data = {}
        
        mapping = {
            'orders': 'olist_orders_dataset',
            'customers': 'olist_customers_dataset',
            'geo': 'olist_geolocation_dataset',
            'items': 'olist_order_items_dataset',
            'products': 'olist_products_dataset',
            'sellers': 'olist_sellers_dataset'
        }
        
        for key, filename in mapping.items():
            data[key] = self.load_csv(filename)
        
        logger.info(f"✅ {len(data)} datasets cargados exitosamente")
        
        return data
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Genera un resumen de los datos cargados
        
        Args:
            data: Diccionario de DataFrames
            
        Returns:
            DataFrame con estadísticas
        """
        summary = []
        
        for name, df in data.items():
            summary.append({
                'Dataset': name,
                'Filas': f"{len(df):,}",
                'Columnas': len(df.columns),
                'Memoria (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            })
        
        return pd.DataFrame(summary)


if __name__ == '__main__':
    loader = OlistDataLoader(
        raw_data_path='data/raw',
        zip_path='data/raw/archive.zip'
    )
    
    loader.extract_zip()
    data = loader.load_all()
    
    print("\n📊 RESUMEN:")
    print(loader.get_data_summary(data))
