import os
from pathlib import Path
"""
Sistema de Versionado y Cache de Modelos - Olist Logistics

Responsabilidad:
    - Guardar modelos con versionado automático
    - Cargar última versión disponible
    - Gestionar cache de resultados
    - Limpiar versiones antiguas
    - Comparar métricas entre versiones

Uso:
    from src.utils.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    
    # Guardar modelo
    registry.save_model(model, "gp_spatial", metrics={'rmse': 40.2})
    
    # Cargar último modelo
    model = registry.load_latest("gp_spatial")
    
    # Limpiar versiones antiguas
    registry.cleanup(keep_last=3)
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Sistema de versionado y gestión de modelos
    """
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            # Auto-detectar ruta del proyecto
            if os.path.exists('/app'):
                base_path = '/app'
            else:
                # Ruta relativa desde este archivo
                # model_registry.py está en: src/utils/
                base_path = str(Path(__file__).parent.parent.parent)
        """
        Args:
            base_path: Ruta base del proyecto
        """
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / 'models'
        self.cache_dir = self.base_path / 'cache'
        
        # Crear estructura
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        (self.models_dir / 'forecasting').mkdir(exist_ok=True)
        (self.models_dir / 'spatial').mkdir(exist_ok=True)
        
        self.metadata_file = self.models_dir / 'metadata.json'
        
        # Cargar o crear metadata
        self._load_metadata()
        
        logger.info(f"ModelRegistry inicializado en {base_path}")
    
    def _load_metadata(self):
        """Carga o crea archivo de metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'forecasting': {},
                'spatial': {},
                'created_at': datetime.now().isoformat()
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Guarda metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, 
                   model: Any, 
                   model_name: str,
                   model_type: str = 'forecasting',
                   metrics: Optional[Dict] = None,
                   overwrite_latest: bool = True) -> str:
        """
        Guarda modelo con versionado automático
        
        Args:
            model: Modelo a guardar
            model_name: Nombre del modelo (ej: 'gp_spatial', 'lgbm_health_beauty')
            model_type: 'forecasting' o 'spatial'
            metrics: Diccionario con métricas (RMSE, R2, etc)
            overwrite_latest: Si actualizar el symlink 'latest'
            
        Returns:
            Ruta donde se guardó el modelo
        """
        # Crear versión con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_name = f"v_{timestamp}"
        
        # Directorio de versión
        version_dir = self.models_dir / model_type / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo
        model_path = version_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        
        logger.info(f"✅ Modelo guardado: {model_path}")
        
        # Guardar métricas
        if metrics:
            metrics_path = version_dir / f"{model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"✅ Métricas guardadas: {metrics_path}")
        
        # Actualizar metadata
        if model_type not in self.metadata:
            self.metadata[model_type] = {}
        
        if model_name not in self.metadata[model_type]:
            self.metadata[model_type][model_name] = []
        
        self.metadata[model_type][model_name].append({
            'version': version_name,
            'timestamp': timestamp,
            'path': str(model_path),
            'metrics': metrics or {}
        })
        
        self._save_metadata()
        
        # Actualizar 'latest'
        if overwrite_latest:
            latest_dir = self.models_dir / model_type / 'latest'
            latest_dir.mkdir(exist_ok=True)
            
            latest_path = latest_dir / f"{model_name}.pkl"
            shutil.copy(model_path, latest_path)
            
            logger.info(f"✅ Latest actualizado: {latest_path}")
        
        return str(model_path)
    
    def load_latest(self, model_name: str, model_type: str = 'forecasting') -> Any:
        """
        Carga última versión del modelo
        
        Args:
            model_name: Nombre del modelo
            model_type: 'forecasting' o 'spatial'
            
        Returns:
            Modelo cargado
        """
        latest_path = self.models_dir / model_type / 'latest' / f"{model_name}.pkl"
        
        if not latest_path.exists():
            raise FileNotFoundError(f"No se encontró modelo: {model_name}")
        
        logger.info(f"📦 Cargando modelo: {latest_path}")
        
        return joblib.load(latest_path)
    
    def load_version(self, model_name: str, version: str, 
                    model_type: str = 'forecasting') -> Any:
        """
        Carga versión específica del modelo
        
        Args:
            model_name: Nombre del modelo
            version: Nombre de la versión (ej: 'v_20260107_123456')
            model_type: 'forecasting' o 'spatial'
            
        Returns:
            Modelo cargado
        """
        model_path = self.models_dir / model_type / version / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Versión no encontrada: {version}")
        
        logger.info(f"📦 Cargando versión: {model_path}")
        
        return joblib.load(model_path)
    
    def list_versions(self, model_name: str, 
                     model_type: str = 'forecasting') -> List[Dict]:
        """
        Lista todas las versiones de un modelo
        
        Args:
            model_name: Nombre del modelo
            model_type: 'forecasting' o 'spatial'
            
        Returns:
            Lista de diccionarios con info de versiones
        """
        if model_type not in self.metadata:
            return []
        
        if model_name not in self.metadata[model_type]:
            return []
        
        return self.metadata[model_type][model_name]
    
    def compare_versions(self, model_name: str, 
                        model_type: str = 'forecasting') -> pd.DataFrame:
        """
        Compara métricas entre versiones
        
        Args:
            model_name: Nombre del modelo
            model_type: 'forecasting' o 'spatial'
            
        Returns:
            DataFrame con comparación
        """
        versions = self.list_versions(model_name, model_type)
        
        if not versions:
            return pd.DataFrame()
        
        data = []
        for v in versions:
            row = {
                'version': v['version'],
                'timestamp': v['timestamp']
            }
            row.update(v['metrics'])
            data.append(row)
        
        return pd.DataFrame(data)
    
    def cleanup(self, keep_last: int = 3, model_type: Optional[str] = None):
        """
        Elimina versiones antiguas
        
        Args:
            keep_last: Cuántas versiones mantener
            model_type: Tipo específico o None para todos
        """
        logger.info(f"🧹 Limpiando versiones antiguas (mantener últimas {keep_last})...")
        
        types_to_clean = [model_type] if model_type else ['forecasting', 'spatial']
        
        for mtype in types_to_clean:
            if mtype not in self.metadata:
                continue
            
            for model_name, versions in self.metadata[mtype].items():
                if len(versions) <= keep_last:
                    continue
                
                # Ordenar por timestamp
                sorted_versions = sorted(versions, 
                                        key=lambda x: x['timestamp'], 
                                        reverse=True)
                
                # Mantener últimas N
                to_keep = sorted_versions[:keep_last]
                to_delete = sorted_versions[keep_last:]
                
                # Eliminar directorios
                for v in to_delete:
                    version_dir = Path(v['path']).parent
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                        logger.info(f"   🗑️ Eliminado: {version_dir.name}")
                
                # Actualizar metadata
                self.metadata[mtype][model_name] = to_keep
        
        self._save_metadata()
        logger.info("✅ Limpieza completada")
    
    def cache_results(self, key: str, data: Any):
        """
        Guarda resultados en cache
        
        Args:
            key: Identificador del cache
            data: Datos a guardar
        """
        cache_path = self.cache_dir / f"{key}.pkl"
        joblib.dump(data, cache_path)
        logger.info(f"💾 Cache guardado: {cache_path}")
    
    def load_cache(self, key: str) -> Optional[Any]:
        """
        Carga resultados desde cache
        
        Args:
            key: Identificador del cache
            
        Returns:
            Datos o None si no existe
        """
        cache_path = self.cache_dir / f"{key}.pkl"
        
        if not cache_path.exists():
            return None
        
        logger.info(f"📦 Cache cargado: {cache_path}")
        return joblib.load(cache_path)
    
    def clear_cache(self):
        """Elimina todo el cache"""
        logger.info("🧹 Limpiando cache...")
        
        for file in self.cache_dir.glob('*.pkl'):
            file.unlink()
            logger.info(f"   🗑️ Eliminado: {file.name}")
        
        logger.info("✅ Cache limpiado")


if __name__ == '__main__':
    print("Módulo model_registry.py cargado")