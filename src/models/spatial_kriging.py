"""
Sistema de Kriging para Interpolación Espacial - Olist Logistics

Responsabilidades:
    - Agregación espacial inteligente
    - Muestreo estratificado
    - Entrenamiento de GP espacial
    - Predicción en malla regular
    - Extracción de insights

Fase del Proyecto: 3 (Inteligencia Espacial)
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialKriging:
    """
    Sistema de Kriging (Gaussian Process espacial)
    
    Features:
        - Agregación espacial por grid
        - Muestreo estratificado
        - Entrenamiento optimizado
        - Predicción en puntos nuevos
        - Incertidumbre espacial
    """
    
    def __init__(self, bounds=None, precision=2):
        """
        Args:
            bounds: Dict con lat_min, lat_max, lng_min, lng_max
            precision: Decimales para redondeo (2 = ~1km)
        """
        if bounds is None:
            bounds = {
                'lat_min': -33.75, 'lat_max': 5.27,
                'lng_min': -73.99, 'lng_max': -34.79
            }
        
        self.bounds = bounds
        self.precision = precision
        self.model = None
        self.df_spatial = None
        self.df_train_geo = None
        
        logger.info(f"SpatialKriging inicializado (precisión: {precision} decimales)")
    
    def aggregate_spatial(self, df, delay_col='delay_days', 
                         lat_col='geolocation_lat', lng_col='geolocation_lng'):
        """
        Agrega datos por coordenadas
        
        Args:
            df: DataFrame con datos originales
            delay_col: Columna de retrasos
            lat_col, lng_col: Columnas de coordenadas
            
        Returns:
            DataFrame agregado
        """
        logger.info("Agregando datos espacialmente...")
        
        df_agg = df.copy()
        df_agg['lat_round'] = df_agg[lat_col].round(self.precision)
        df_agg['lng_round'] = df_agg[lng_col].round(self.precision)
        
        # Validar bounds
        df_agg = df_agg[
            (df_agg['lat_round'].between(self.bounds['lat_min'], self.bounds['lat_max'])) &
            (df_agg['lng_round'].between(self.bounds['lng_min'], self.bounds['lng_max']))
        ]
        
        # Agrupar
        self.df_spatial = df_agg.groupby(['lat_round', 'lng_round']).agg({
            delay_col: 'mean',
            'order_id': 'count'
        }).reset_index().rename(columns={
            delay_col: 'retraso_promedio',
            'order_id': 'volumen_pedidos'
        })
        
        logger.info(f"   Ubicaciones únicas: {len(self.df_spatial):,}")
        
        return self.df_spatial
    
    def stratified_sampling(self, n_target=4000):
        """
        Muestreo estratificado: 100% retrasos + contexto
        
        Args:
            n_target: Total de puntos objetivo
            
        Returns:
            DataFrame de entrenamiento
        """
        if self.df_spatial is None:
            raise ValueError("Ejecuta aggregate_spatial() primero")
        
        logger.info("Aplicando muestreo estratificado...")
        
        # Separar señal vs ruido
        df_retrasos = self.df_spatial[self.df_spatial['retraso_promedio'] > 0]
        df_normales = self.df_spatial[self.df_spatial['retraso_promedio'] <= 0]
        
        logger.info(f"   Puntos con retraso: {len(df_retrasos):,}")
        
        n_faltantes = n_target - len(df_retrasos)
        
        if n_faltantes > 0:
            df_normal_sample = df_normales.sample(n_faltantes, random_state=42)
            self.df_train_geo = pd.concat([df_retrasos, df_normal_sample])
        else:
            self.df_train_geo = df_retrasos.sample(n_target, random_state=42)
        
        # Shuffle
        self.df_train_geo = self.df_train_geo.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"   Dataset entrenamiento: {len(self.df_train_geo):,} puntos")
        
        return self.df_train_geo
    
    def train(self, length_scale_bounds=(0.1, 10.0)):
        """
        Entrena GP espacial
        
        Args:
            length_scale_bounds: Rango de búsqueda para length_scale
        """
        if self.df_train_geo is None:
            raise ValueError("Ejecuta stratified_sampling() primero")
        
        logger.info("Entrenando modelo Kriging...")
        
        # Preparar datos
        X_geo = self.df_train_geo[['lat_round', 'lng_round']].values
        y_geo = self.df_train_geo['retraso_promedio'].values
        
        gc.collect()
        
        # Kernel
        k_spatial = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds) + \
                    WhiteKernel(noise_level=0.1)
        
        # Entrenar
        self.model = GaussianProcessRegressor(
            kernel=k_spatial,
            normalize_y=True,
            random_state=42,
            copy_X_train=False
        )
        
        self.model.fit(X_geo, y_geo)
        
        # Métricas
        r2 = self.model.score(X_geo, y_geo)
        length_scale = self.model.kernel_.get_params()['k1__k2__length_scale']
        
        logger.info(f"   ✅ R²: {r2:.3f}")
        logger.info(f"   ✅ Length Scale: {length_scale:.2f}° (~{length_scale * 111:.0f} km)")
        
        return {
            'r2': r2,
            'length_scale': length_scale,
            'length_scale_km': length_scale * 111
        }
    
    def predict(self, X_new, return_std=False):
        """
        Predice en nuevos puntos
        
        Args:
            X_new: Array (n, 2) con [lat, lng]
            return_std: Si devolver desviación estándar
            
        Returns:
            Array con predicciones (y std si solicitado)
        """
        if self.model is None:
            raise ValueError("Entrena el modelo primero")
        
        return self.model.predict(X_new, return_std=return_std)
    
    def predict_grid(self, res_lat=300):
        """
        Genera predicciones en malla regular
        
        Args:
            res_lat: Resolución vertical
            
        Returns:
            Dict con grid_lat, grid_lng, predictions, std
        """
        if self.model is None:
            raise ValueError("Entrena el modelo primero")
        
        logger.info("Generando malla de predicción...")
        
        # Calcular aspecto
        delta_lat = self.bounds['lat_max'] - self.bounds['lat_min']
        delta_lng = self.bounds['lng_max'] - self.bounds['lng_min']
        aspect_ratio = delta_lng / delta_lat
        
        res_lng = int(res_lat * aspect_ratio)
        
        # Crear malla
        grid_lat = np.linspace(self.bounds['lat_min'], self.bounds['lat_max'], res_lat)
        grid_lng = np.linspace(self.bounds['lng_min'], self.bounds['lng_max'], res_lng)
        
        xx, yy = np.meshgrid(grid_lng, grid_lat)
        X_grid_flat = np.column_stack((yy.ravel(), xx.ravel()))
        
        logger.info(f"   Prediciendo en {len(X_grid_flat):,} puntos...")
        
        # Predecir
        y_pred_flat, sigma_flat = self.model.predict(X_grid_flat, return_std=True)
        
        # Reconstruir matrices
        Z = y_pred_flat.reshape(res_lat, res_lng)
        Sigma = sigma_flat.reshape(res_lat, res_lng)
        
        logger.info("   ✅ Malla generada")
        
        return {
            'grid_lat': grid_lat,
            'grid_lng': grid_lng,
            'predictions': Z,
            'std': Sigma,
            'xx': xx,
            'yy': yy
        }
    
    def get_insights(self):
        """Extrae parámetros del modelo"""
        if self.model is None:
            return {}
        
        length_scale = self.model.kernel_.get_params()['k1__k2__length_scale']
        
        return {
            'length_scale': length_scale,
            'length_scale_km': length_scale * 111,
            'n_training': len(self.df_train_geo),
            'n_delay_points': len(self.df_spatial[self.df_spatial['retraso_promedio'] > 0])
        }


if __name__ == '__main__':
    print("Módulo spatial_kriging.py cargado")
