"""
Análisis Espacial Avanzado - Olist Logistics Intelligence

Responsabilidades:
    - Generación de mallas de predicción
    - Máscaras geográficas
    - Mapas maestros con múltiples capas
    - Análisis de outliers espaciales
    - Calibración de transparencia por incertidumbre

Fase del Proyecto: 3 (Análisis Espacial Avanzado)

Uso:
    from src.visualization.spatial_analytics import SpatialAnalytics
    
    analytics = SpatialAnalytics(gp_model)
    mapa = analytics.create_master_map(
        df_train_geo=df_train_geo,
        save_path='outputs/mapa_maestro.html'
    )
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point
from shapely.prepared import prep
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialAnalytics:
    """
    Sistema de análisis espacial avanzado con visualizaciones maestras
    
    Features:
        - Generación de mallas de alta resolución
        - Máscaras geográficas precisas
        - Alpha mapping por incertidumbre
        - Detección de outliers espaciales
        - Mapas interactivos multicapa
    """
    
    def __init__(self, gp_model, bounds=None):
        """
        Args:
            gp_model: Modelo GP entrenado (sklearn GaussianProcessRegressor)
            bounds: Dict con lat_min, lat_max, lng_min, lng_max
        """
        if bounds is None:
            bounds = {
                'lat_min': -33.75, 'lat_max': 5.27,
                'lng_min': -73.99, 'lng_max': -34.79
            }
        
        self.gp_model = gp_model
        self.bounds = bounds
        self.gdf_brazil = None
        self.prep_brazil = None
        
        logger.info("SpatialAnalytics inicializado")
    
    def load_brazil_geometry(self, url=None):
        """
        Carga geometría de Brasil desde GeoJSON
        
        Args:
            url: URL del GeoJSON (None = usar default)
        """
        if url is None:
            url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
        
        logger.info("Cargando geometría de Brasil...")
        
        self.gdf_brazil = gpd.read_file(url)[['geometry']]
        
        # Unión de todas las geometrías
        try:
            pais_brazil = self.gdf_brazil.union_all()
        except AttributeError:
            pais_brazil = self.gdf_brazil.unary_union
        
        # Preparar para queries rápidas
        self.prep_brazil = prep(pais_brazil)
        
        logger.info("✅ Geometría cargada")
    
    def generate_prediction_grid(self, res_lat=300) -> Dict:
        """
        Genera malla de predicción con aspecto correcto
        
        Args:
            res_lat: Resolución vertical (filas)
            
        Returns:
            Dict con grid_lat, grid_lng, xx, yy, X_grid_flat
        """
        logger.info(f"Generando malla de predicción (res={res_lat})...")
        
        # Calcular aspecto
        delta_lat = self.bounds['lat_max'] - self.bounds['lat_min']
        delta_lng = self.bounds['lng_max'] - self.bounds['lng_min']
        aspect_ratio = delta_lng / delta_lat
        
        res_lng = int(res_lat * aspect_ratio)
        
        logger.info(f"   Dimensiones: {res_lat}×{res_lng} (Aspect: {aspect_ratio:.2f})")
        
        # Crear malla
        grid_lat = np.linspace(self.bounds['lat_min'], self.bounds['lat_max'], res_lat)
        grid_lng = np.linspace(self.bounds['lng_min'], self.bounds['lng_max'], res_lng)
        
        xx, yy = np.meshgrid(grid_lng, grid_lat)
        X_grid_flat = np.column_stack((yy.ravel(), xx.ravel()))
        
        logger.info(f"   Total píxeles: {len(X_grid_flat):,}")
        
        return {
            'grid_lat': grid_lat,
            'grid_lng': grid_lng,
            'xx': xx,
            'yy': yy,
            'X_grid_flat': X_grid_flat,
            'res_lat': res_lat,
            'res_lng': res_lng
        }
    
    def predict_on_grid(self, grid_result: Dict) -> Dict:
        """
        Predice en toda la malla
        
        Args:
            grid_result: Output de generate_prediction_grid()
            
        Returns:
            Dict con Z (predicciones) y Sigma (incertidumbre)
        """
        logger.info("Calculando predicciones...")
        
        import time
        start = time.time()
        
        y_pred_flat, sigma_flat = self.gp_model.predict(
            grid_result['X_grid_flat'], 
            return_std=True
        )
        
        elapsed = time.time() - start
        logger.info(f"✅ Predicción completa en {elapsed:.1f}s")
        
        # Reconstruir matrices
        Z = y_pred_flat.reshape(grid_result['res_lat'], grid_result['res_lng'])
        Sigma = sigma_flat.reshape(grid_result['res_lat'], grid_result['res_lng'])
        
        return {'Z': Z, 'Sigma': Sigma}
    
    def apply_geographic_mask(self, grid_result: Dict, pred_result: Dict) -> np.ma.MaskedArray:
        """
        Aplica máscara geográfica (solo Brasil)
        
        Args:
            grid_result: Output de generate_prediction_grid()
            pred_result: Output de predict_on_grid()
            
        Returns:
            Matriz Z enmascarada
        """
        if self.prep_brazil is None:
            self.load_brazil_geometry()
        
        logger.info("Aplicando máscara geográfica...")
        
        Z = pred_result['Z']
        xx = grid_result['xx']
        yy = grid_result['yy']
        
        mask_geo = np.zeros_like(Z, dtype=bool)
        
        for i in range(grid_result['res_lat']):
            for j in range(grid_result['res_lng']):
                if not self.prep_brazil.contains(Point(xx[i, j], yy[i, j])):
                    mask_geo[i, j] = True
        
        Z_masked = np.ma.masked_where(mask_geo, Z)
        
        logger.info("✅ Máscara aplicada")
        
        return Z_masked, mask_geo
    
    def generate_rgba_image(self, Z_masked, Sigma, mask_geo) -> Tuple[np.ndarray, float, float]:
        """
        Genera imagen RGBA con alpha mapping
        
        Args:
            Z_masked: Matriz de predicciones enmascarada
            Sigma: Matriz de incertidumbre
            mask_geo: Máscara booleana
            
        Returns:
            (image_final, min_val, max_val)
        """
        logger.info("Generando imagen RGBA...")
        
        # Alpha map (transparencia por incertidumbre)
        sigma_min = Sigma.min()
        sigma_max = np.percentile(Sigma, 95)
        alpha_map = 1.0 - ((Sigma - sigma_min) / (sigma_max - sigma_min))
        alpha_map = np.clip(alpha_map, 0, 1) ** 1.5
        alpha_map[mask_geo] = 0
        
        # Normalización de colores
        datos_validos = Z_masked.compressed()
        min_val = np.percentile(datos_validos, 5)
        max_val = np.percentile(datos_validos, 95)
        
        logger.info(f"   Rango: [{min_val:.1f}, {max_val:.1f}] días")
        
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        try:
            cmap = plt.colormaps.get_cmap('turbo')
        except:
            cmap = plt.cm.get_cmap('jet')
        
        # Crear imagen RGBA
        image_data = cmap(norm(Z_masked))
        image_data[:, :, 3] = alpha_map
        
        # Flip vertical (coordenadas vs matriz)
        image_final = np.flipud(image_data)
        
        logger.info("✅ Imagen generada")
        
        return image_final, min_val, max_val
    
    def create_master_map(self, 
                         df_train_geo: pd.DataFrame,
                         res_lat: int = 300,
                         sample_points: int = 1000,
                         save_path: Optional[str] = None) -> folium.Map:
        """
        Genera mapa maestro completo
        
        Args:
            df_train_geo: DataFrame con puntos de entrenamiento
            res_lat: Resolución de la malla
            sample_points: Cuántos puntos mostrar (None = todos)
            save_path: Ruta para guardar HTML
            
        Returns:
            Objeto folium.Map
        """
        logger.info("="*70)
        logger.info("GENERANDO MAPA MAESTRO")
        logger.info("="*70)
        
        # Paso 1: Cargar geometría
        if self.prep_brazil is None:
            self.load_brazil_geometry()
        
        # Paso 2: Generar malla
        grid_result = self.generate_prediction_grid(res_lat=res_lat)
        
        # Paso 3: Predecir
        pred_result = self.predict_on_grid(grid_result)
        
        # Paso 4: Máscara
        Z_masked, mask_geo = self.apply_geographic_mask(grid_result, pred_result)
        
        # Paso 5: Imagen RGBA
        image_final, min_val, max_val = self.generate_rgba_image(
            Z_masked, pred_result['Sigma'], mask_geo
        )
        
        # Paso 6: Crear mapa base
        logger.info("Construyendo mapa interactivo...")
        
        mapa = folium.Map(
            location=[-15.79, -47.88],
            zoom_start=4,
            tiles='cartodbpositron'
        )
        
        # Capa 1: Predicción
        folium.raster_layers.ImageOverlay(
            image=image_final,
            bounds=[
                [self.bounds['lat_min'], self.bounds['lng_min']],
                [self.bounds['lat_max'], self.bounds['lng_max']]
            ],
            opacity=0.8,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(mapa)
        
        # Capa 2: Fronteras
        folium.GeoJson(
            self.gdf_brazil,
            name='Fronteras',
            style_function=lambda x: {
                'color': 'black',
                'weight': 0.8,
                'fillOpacity': 0
            }
        ).add_to(mapa)
        
        # Capa 3: Puntos de auditoría
        logger.info("Añadiendo puntos de auditoría...")
        
        grupo_puntos = folium.FeatureGroup(name="Datos Reales (Auditoría)", zindex=100)
        
        # Muestrear si es necesario
        if sample_points and len(df_train_geo) > sample_points:
            df_sample = df_train_geo.sample(sample_points, random_state=42)
        else:
            df_sample = df_train_geo
        
        for idx, row in df_sample.iterrows():
            val = row['retraso_promedio']
            es_critico = val > max_val * 0.8
            
            folium.CircleMarker(
                location=[row['lat_round'], row['lng_round']],
                radius=4 if es_critico else 2,
                color='white',
                weight=1,
                fill=True,
                fill_color='#FF00FF' if es_critico else 'black',
                fill_opacity=1.0,
                popup=f"Real: {val:.1f} días"
            ).add_to(grupo_puntos)
        
        grupo_puntos.add_to(mapa)
        
        logger.info(f"✅ {len(df_sample):,} puntos añadidos")
        
        # Leyenda
        leyenda_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
             background: rgba(255, 255, 255, 0.95); padding: 15px; 
             border-radius: 8px; border: 1px solid #999; 
             font-family: sans-serif; box-shadow: 4px 4px 10px rgba(0,0,0,0.2); 
             width: 220px;">
             
             <h4 style="margin:0 0 10px 0; color:#333; 
                        border-bottom:1px solid #ccc; padding-bottom:5px;">
                📡 Inteligencia Logística
             </h4>
             
             <b style="font-size:11px; color:#555;">PREDICCIÓN (FONDO)</b>
             <div style="background: linear-gradient(to right, 
                         #30123b, #28bbec, #a2fc3c, #fb8022, #7a0403); 
                         height: 10px; width: 100%; margin-top:5px; 
                         border-radius:2px;"></div>
             <div style="display: flex; justify-content: space-between; 
                         font-size: 10px; color: #333; margin-bottom:5px;">
                <span>{min_val:.1f}d</span>
                <span>{max_val:.1f}d</span>
             </div>
             
             <div style="background: linear-gradient(to right, 
                         rgba(0,0,0,1), rgba(0,0,0,0)); 
                         height: 6px; width: 100%; border: 1px solid #eee;"></div>
             <div style="display: flex; justify-content: space-between; 
                         font-size: 9px; color: #777; margin-bottom:10px;">
                <span>Certeza Alta</span>
                <span>Niebla</span>
             </div>
             
             <b style="font-size:11px; color:#555;">AUDITORÍA</b>
             <div style="margin-top:5px;">
                <span style="color:#FF00FF;">●</span> Crítico (> {max_val*0.8:.1f}d)<br>
                <span style="color:black;">●</span> Normal
             </div>
             
             <div style="margin-top:10px; padding-top:10px; 
                         border-top:1px solid #eee; font-size:9px; color:#666;">
                <b>Nota:</b> Puntos magenta en zonas verdes = 
                Outliers locales (eventos aislados, no sistémicos)
             </div>
        </div>
        """
        
        mapa.get_root().html.add_child(folium.Element(leyenda_html))
        folium.LayerControl(collapsed=False).add_to(mapa)
        
        # Guardar
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            mapa.save(save_path)
            logger.info(f"💾 Guardado: {save_path}")
        
        logger.info("="*70)
        logger.info("✅ MAPA MAESTRO COMPLETADO")
        logger.info("="*70)
        
        # Estadísticas
        length_scale = self.gp_model.kernel_.get_params()['k1__k2__length_scale']
        n_critical = len(df_train_geo[df_train_geo['retraso_promedio'] > max_val * 0.8])
        
        logger.info(f"Radio influencia: ~{length_scale * 111:.0f} km")
        logger.info(f"Puntos críticos: {n_critical:,}")
        logger.info(f"Rango predicho: [{min_val:.1f}, {max_val:.1f}] días")
        
        return mapa


if __name__ == '__main__':
    print("Módulo spatial_analytics.py cargado")
