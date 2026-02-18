"""
Visualizaciones Espaciales - Olist Logistics

Responsabilidades:
    - Mapa de calor histórico
    - Mapa de predicciones
    - Mapa de incertidumbre
    - Interfaz de cálculo de envíos

Fase del Proyecto: 3 (Visualización Espacial)
"""

import folium
from folium.plugins import HeatMap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialVisualizer:
    """
    Generador de visualizaciones espaciales
    """
    
    def __init__(self, center=None, zoom_start=4):
        """
        Args:
            center: [lat, lng] del centro del mapa
            zoom_start: Nivel de zoom inicial
        """
        if center is None:
            center = [-14.2350, -51.9253]  # Brasil
        
        self.center = center
        self.zoom_start = zoom_start
    
    def create_heatmap(self, df_spatial, value_col='retraso_promedio',
                      lat_col='lat_round', lng_col='lng_round',
                      save_path=None):
        """
        Crea mapa de calor histórico
        
        Args:
            df_spatial: DataFrame con datos agregados
            value_col: Columna de valores
            lat_col, lng_col: Columnas de coordenadas
            save_path: Ruta para guardar HTML
            
        Returns:
            Objeto folium.Map
        """
        logger.info("Creando mapa de calor...")
        
        # Filtrar solo retrasos
        df_risk = df_spatial[df_spatial[value_col] > 0].copy()
        
        logger.info(f"   Zonas con retraso: {len(df_risk):,}")
        
        # Crear mapa
        mapa = folium.Map(location=self.center, zoom_start=self.zoom_start, tiles='cartodbpositron')
        
        # Datos
        datos_calor = df_risk[[lat_col, lng_col, value_col]].values.tolist()
        
        HeatMap(
            datos_calor,
            radius=15,
            blur=20,
            max_zoom=1,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(mapa)
        
        # Leyenda
        leyenda = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 300px;
                    background: white; border: 2px solid grey; padding: 10px;
                    z-index: 9999; opacity: 0.9;">
            <b>Mapa de Riesgo Logístico (Histórico)</b><br>
            <b style="color:red">Rojo:</b> Zonas con alto retraso<br>
            <b style="color:lime">Verde:</b> Retrasos menores
        </div>
        """
        mapa.get_root().html.add_child(folium.Element(leyenda))
        
        if save_path:
            mapa.save(save_path)
            logger.info(f"   Guardado: {save_path}")
        
        return mapa
    
    def create_prediction_map(self, grid_result, gdf_brazil=None,
                             df_train_geo=None, save_path=None):
        """
        Crea mapa maestro con predicciones
        
        Args:
            grid_result: Dict con resultados de predict_grid()
            gdf_brazil: GeoDataFrame con geometría de Brasil
            df_train_geo: DataFrame con puntos de entrenamiento
            save_path: Ruta para guardar
            
        Returns:
            Objeto folium.Map
        """
        logger.info("Creando mapa de predicciones...")
        
        # Extraer datos
        Z = grid_result['predictions']
        Sigma = grid_result['std']
        
        # Calcular límites
        datos_validos = Z[~np.isnan(Z)]
        min_val = np.percentile(datos_validos, 5)
        max_val = np.percentile(datos_validos, 95)
        
        # Normalizar
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = plt.colormaps.get_cmap('turbo')
        
        # Alpha map (incertidumbre)
        sigma_min = Sigma.min()
        sigma_max = np.percentile(Sigma, 95)
        alpha_map = 1.0 - ((Sigma - sigma_min) / (sigma_max - sigma_min))
        alpha_map = np.clip(alpha_map, 0, 1) ** 1.5
        
        # Imagen RGBA
        image_data = cmap(norm(Z))
        image_data[:, :, 3] = alpha_map
        image_final = np.flipud(image_data)
        
        # Crear mapa
        mapa = folium.Map(location=self.center, zoom_start=self.zoom_start, tiles='cartodbpositron')
        
        # Overlay de predicción
        bounds = [
            [grid_result['grid_lat'].min(), grid_result['grid_lng'].min()],
            [grid_result['grid_lat'].max(), grid_result['grid_lng'].max()]
        ]
        
        folium.raster_layers.ImageOverlay(
            image=image_final,
            bounds=bounds,
            opacity=0.8,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(mapa)
        
        # Fronteras
        if gdf_brazil is not None:
            folium.GeoJson(
                gdf_brazil,
                name='Fronteras',
                style_function=lambda x: {'color': 'black', 'weight': 0.8, 'fillOpacity': 0}
            ).add_to(mapa)
        
        # Puntos de entrenamiento
        if df_train_geo is not None:
            logger.info("   Añadiendo puntos de auditoría...")
            grupo = folium.FeatureGroup(name="Datos Reales", zindex=100)
            
            for _, row in df_train_geo.iterrows():
                val = row['retraso_promedio']
                critico = val > max_val * 0.8
                
                folium.CircleMarker(
                    location=[row['lat_round'], row['lng_round']],
                    radius=4 if critico else 2,
                    color='white',
                    weight=1,
                    fill=True,
                    fill_color='#FF00FF' if critico else 'black',
                    fill_opacity=1.0,
                    popup=f"Real: {val:.1f} días"
                ).add_to(grupo)
            
            grupo.add_to(mapa)
        
        # Leyenda
        leyenda = f"""
        <div style="position: fixed; bottom: 30px; left: 30px;
                    background: rgba(255,255,255,0.95); padding: 15px;
                    border: 1px solid #999; border-radius: 8px;
                    z-index: 9999; width: 220px; font-family: sans-serif;">
            <h4 style="margin: 0 0 10px 0;">📡 Inteligencia Logística</h4>
            <b style="font-size: 11px;">PREDICCIÓN</b>
            <div style="background: linear-gradient(to right, #30123b, #28bbec, #a2fc3c, #fb8022, #7a0403);
                        height: 10px; margin: 5px 0; border-radius: 2px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                <span>{min_val:.1f}d</span>
                <span>{max_val:.1f}d</span>
            </div>
            <div style="margin-top: 10px;">
                <span style="color: #FF00FF;">●</span> Crítico (>{max_val*0.8:.1f}d)
            </div>
        </div>
        """
        mapa.get_root().html.add_child(folium.Element(leyenda))
        
        folium.LayerControl(collapsed=False).add_to(mapa)
        
        if save_path:
            mapa.save(save_path)
            logger.info(f"   Guardado: {save_path}")
        
        return mapa


if __name__ == '__main__':
    print("Módulo spatial_viz.py cargado")
