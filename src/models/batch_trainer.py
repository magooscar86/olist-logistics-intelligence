"""
Módulo de Entrenamiento Batch - Olist Logistics Intelligence

Responsabilidad:
    - Entrenar modelos GP para múltiples categorías
    - Traducción de nombres portugués → inglés
    - Detección automática de categorías rápidas
    - Registro de modelos entrenados (model_registry)

Fase del Proyecto: 2 (Forecasting)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from src.utils.model_registry import ModelRegistry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchTrainer:
    """
    Entrena modelos GP para múltiples categorías en paralelo
    
    Capacidades:
        - Selección automática de Top N categorías
        - Traducción portugués/inglés
        - Detección de categorías rápidas (usa config FAST)
        - Validación de datos mínimos
        - Registro estructurado de resultados
    
    Example:
        >>> from src.models.batch_trainer import BatchTrainer
        >>> trainer = BatchTrainer()
        >>> registry = trainer.train_batch(df_main, top_n=6)
        >>> print(registry.keys())  # Categorías entrenadas
    """
    
    # Diccionario de traducción portugués → inglés
    TRANSLATION = {
        'cama_mesa_banho': 'bed_bath_table',
        'beleza_saude': 'health_beauty',
        'esporte_lazer': 'sports_leisure',
        'moveis_decoracao': 'furniture_decor',
        'informatica_acessorios': 'computers_accessories',
        'utilidades_domesticas': 'housewares',
        'relogios_presentes': 'watches_gifts',
        'telefonia': 'telephony',
        'ferramentas_jardim': 'garden_tools',
        'automotivo': 'auto',
        'brinquedos': 'toys',
        'cool_stuff': 'cool_stuff',
        'perfumaria': 'perfumery',
        'bebes': 'baby',
        'eletronicos': 'electronics'
    }
    
    def __init__(self):
        """Inicializa el batch trainer"""
        from config import TIME_SERIES_CONFIG, FAST_ROTATION_CATEGORIES
        
        self.ts_config = TIME_SERIES_CONFIG
        self.fast_categories = FAST_ROTATION_CATEGORIES
        
        logger.info("🏭 BatchTrainer inicializado")
    
    def translate_category(self, category_pt: str) -> str:
        """
        Traduce nombre de categoría de portugués a inglés
        
        Args:
            category_pt: Nombre en portugués
            
        Returns:
            Nombre en inglés (o mismo nombre si no hay traducción)
        """
        return self.TRANSLATION.get(category_pt, category_pt)
    
    def get_top_categories(self, 
                          df: pd.DataFrame,
                          top_n: int = 6,
                          category_column: str = 'product_category_name') -> List[str]:
        """
        Identifica las Top N categorías por volumen
        
        Args:
            df: DataFrame con datos
            top_n: Número de categorías a seleccionar
            category_column: Nombre de la columna de categorías
            
        Returns:
            Lista de nombres de categorías (portugués)
        """
        if category_column not in df.columns:
            raise ValueError(f"Columna '{category_column}' no existe en el DataFrame")
        
        # Filtrar Unknown/NaN
        df_valid = df[df[category_column].notna() & (df[category_column] != 'Unknown')]
        
        # Contar por categoría
        top_cats = df_valid[category_column].value_counts().head(top_n).index.tolist()
        
        logger.info(f"🏆 Top {top_n} categorías seleccionadas:")
        for i, cat in enumerate(top_cats, 1):
            count = (df[category_column] == cat).sum()
            cat_en = self.translate_category(cat)
            logger.info(f"   {i}. {cat_en} ({cat}): {count:,} registros")
        
        return top_cats
    
    def is_fast_rotation(self, category_en: str) -> bool:
        """
        Detecta si una categoría es de rotación rápida
        
        Args:
            category_en: Nombre en inglés
            
        Returns:
            True si es categoría rápida
        """
        return category_en in self.fast_categories
    
    def train_category(self,
                      df: pd.DataFrame,
                      category_pt: str,
                      processor,
                      min_weeks: int = 10) -> Optional[Dict]:
        """
        Entrena modelo para una categoría específica
        
        Args:
            df: DataFrame completo
            category_pt: Nombre de categoría (portugués)
            processor: Instancia de TimeSeriesProcessor
            min_weeks: Mínimo de semanas requeridas
            
        Returns:
            Dict con modelo y metadata, o None si falla
        """
        from src.models.gaussian_process_model import GaussianProcessForecaster
        
        category_en = self.translate_category(category_pt)
        
        logger.info(f"\n📦 [{category_en}] Entrenando...")
        
        try:
            # 1. Preparar datos
            X, y, dates = processor.process(
                df, 
                category=category_pt,
                validate=True
            )
            
            # 2. Detectar si es rápida
            use_fast = self.is_fast_rotation(category_en)
            
            if use_fast:
                logger.info(f"   ⚡ Categoría rápida detectada → Usando GP_KERNEL_CONFIG_FAST")
            
            # 3. Crear y entrenar modelo
            model = GaussianProcessForecaster(use_fast_config=use_fast)
            model.train(X, y)
            
            # 4. Extraer insights
            insights = model.get_insights()
            memoria = insights.get('memoria_predictiva', 0)
            
            # 5. Clasificar
            if memoria >= 40:
                tipo = "ESTRATÉGICA"
            elif memoria >= 20:
                tipo = "TÁCTICA"
            else:
                tipo = "VOLÁTIL"
            
            logger.info(f"   ✅ Memoria: {memoria:.1f} sem → {tipo}")
            
            # 6. Crear registro
            result = {
                'model': model,
                'X_train': X,
                'y_train': y,
                'dates': dates,
                'memory': memoria,
                'config_used': 'FAST' if use_fast else 'STANDARD',
                'classification': tipo,
                'insights': insights,
                'n_weeks': len(y),
                'avg_demand': float(y.mean())
            }
            
            return result
            
        except ValueError as e:
            logger.warning(f"   ⚠️  Omitiendo: {e}")
            return None
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            return None
    
    def train_batch(self,
                   df: pd.DataFrame,
                   top_n: int = 6,
                   min_weeks: int = 10,
                   category_column: str = 'product_category_name') -> Dict[str, Dict]:
        """
        Entrena modelos para múltiples categorías
        
        Args:
            df: DataFrame con datos
            top_n: Número de categorías a entrenar
            min_weeks: Mínimo de semanas para validar
            category_column: Nombre de columna de categorías
            
        Returns:
            model_registry: Dict con modelos entrenados
            
        Example:
            >>> trainer = BatchTrainer()
            >>> registry = trainer.train_batch(df_main, top_n=6)
            >>> model = registry['bed_bath_table']['model']
        """
        from src.models.time_series_processor import TimeSeriesProcessor
        
        logger.info("🚀 INICIANDO ENTRENAMIENTO BATCH")
        logger.info("="*70)
        
        # 1. Seleccionar Top N
        top_categories_pt = self.get_top_categories(df, top_n, category_column)
        
        # 2. Crear processor
        processor = TimeSeriesProcessor(
            start_date=self.ts_config['start_date'],
            end_date=self.ts_config['end_date'],
            min_weeks=min_weeks
        )
        
        # 3. Entrenar cada categoría
        model_registry = {}
        
        for i, cat_pt in enumerate(top_categories_pt, 1):
            cat_en = self.translate_category(cat_pt)
            
            logger.info(f"\n[{i}/{len(top_categories_pt)}] Procesando: {cat_en}")
            
            result = self.train_category(df, cat_pt, processor, min_weeks)
            
            if result is not None:
                model_registry[cat_en] = result
        
        # 4. Resumen final
        logger.info("\n" + "="*70)
        logger.info("🎉 ENTRENAMIENTO BATCH COMPLETADO")
        logger.info("="*70)
        
        logger.info(f"\n✅ Modelos entrenados: {len(model_registry)}/{len(top_categories_pt)}")
        
        return model_registry
    
    def get_summary_table(self, model_registry: Dict[str, Dict]) -> pd.DataFrame:
        """
        Genera tabla resumen de modelos entrenados
        
        Args:
            model_registry: Registro de modelos
            
        Returns:
            DataFrame con resumen
        """
        data = []
        
        for category, info in model_registry.items():
            data.append({
                'Categoría': category,
                'Memoria (sem)': round(info['memory'], 1),
                'Clasificación': info['classification'],
                'Config': info['config_used'],
                'Semanas': info['n_weeks'],
                'Demanda Prom': round(info['avg_demand'], 1)
            })
        
        df_summary = pd.DataFrame(data)
        df_summary = df_summary.sort_values('Memoria (sem)', ascending=False)
        
        return df_summary



    def save_models(self, registry=None):
        """
        Guarda todos los modelos entrenados en el registry
        
        Args:
            registry: ModelRegistry instance (se crea si es None)
        """
        if registry is None:
            registry = ModelRegistry()
        
        logger.info("Guardando modelos en registry...")
        
        for cat, results in self.results.items():
            for model_name, model_obj in results['models'].items():
                if model_obj is not None:
                    # Extraer métricas
                    metrics = {
                        'rmse': results['rmse'].get(model_name),
                        'category': cat
                    }
                    
                    # Nombre único
                    full_name = f"{model_name}_{cat.replace(' ', '_')}"
                    
                    # Guardar
                    registry.save_model(
                        model=model_obj,
                        model_name=full_name,
                        model_type='forecasting',
                        metrics=metrics
                    )
        
        logger.info("✅ Modelos guardados en registry")

# ============================================================================
# TESTING
# ============================================================================
if __name__ == '__main__':
    print("Módulo batch_trainer.py cargado correctamente")
