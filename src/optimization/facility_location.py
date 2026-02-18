"""
Optimización de Ubicación de Instalaciones - Olist Logistics
"""

import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacilityLocator:
    """Sistema de optimización de ubicación de instalaciones"""
    
    def __init__(self, df_main: pd.DataFrame):
        self.df_main = df_main
        logger.info("FacilityLocator inicializado")
    
    def optimize_locations(self, n_hubs: int = 4, weight_power: float = 2.0) -> Dict:
        """Encuentra ubicaciones óptimas para hubs"""
        logger.info(f"Optimizando ubicación de {n_hubs} hubs...")
        
        df_pain = self.df_main[self.df_main['delay_days'] > 0].copy()
        logger.info(f"   Analizando {len(df_pain):,} incidentes logísticos")
        
        X_pain = df_pain[['geolocation_lat', 'geolocation_lng']].values
        weights = df_pain['delay_days'].values ** weight_power
        
        kmeans = KMeans(n_clusters=n_hubs, random_state=42, n_init=10)
        kmeans.fit(X_pain, sample_weight=weights)
        
        hub_coords = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        hubs = []
        for i, (lat, lng) in enumerate(hub_coords):
            city, state = self._get_nearest_city(lat, lng)
            
            mask_cluster = labels == i
            carga = np.sum(mask_cluster)
            avg_delay = df_pain.loc[mask_cluster, 'delay_days'].mean()
            
            hubs.append({
                'hub_id': i + 1,
                'lat': lat,
                'lng': lng,
                'city': city,
                'state': state,
                'load': carga,
                'avg_delay': avg_delay
            })
            
            logger.info(f"   Hub {i+1}: {city} ({state}) - {carga:,} incidentes")
        
        return {'hubs': hubs, 'n_incidents': len(df_pain), 'model': kmeans}
    
    def _get_nearest_city(self, lat: float, lng: float) -> Tuple[str, str]:
        """Encuentra ciudad más cercana"""
        dists = (self.df_main['geolocation_lat'] - lat)**2 +                 (self.df_main['geolocation_lng'] - lng)**2
        
        idx = dists.idxmin()
        city = self.df_main.loc[idx, 'geolocation_city'].title()
        state = self.df_main.loc[idx, 'geolocation_state']
        
        return city, state
    
    def visualize_hubs(self, results: Dict, base_map=None, 
                      coverage_radius_km: int = 300,
                      show_risk_heatmap: bool = True):
        """
        Visualiza hubs en mapa interactivo con contexto completo
        
        Args:
            results: Output de optimize_locations()
            base_map: Mapa base opcional (default: nuevo mapa)
            coverage_radius_km: Radio de cobertura en km
            show_risk_heatmap: Si mostrar mapa de calor de fondo
            
        Returns:
            Mapa folium con hubs
        """
        logger.info("Generando visualización de hubs con contexto...")
        
        # Crear mapa base si no se proporciona
        if base_map is None:
            mapa = folium.Map(
                location=[-15.79, -47.88],
                zoom_start=4,
                tiles='cartodbpositron'
            )
        else:
            mapa = base_map
        
        # AÑADIR MAPA DE CALOR DE RIESGO (contexto)
        if show_risk_heatmap:
            logger.info("   Añadiendo mapa de calor de riesgo...")
            
            # Crear agregación espacial
            df_geo_agg = self.df_main.copy()
            df_geo_agg['lat_round'] = df_geo_agg['geolocation_lat'].round(2)
            df_geo_agg['lng_round'] = df_geo_agg['geolocation_lng'].round(2)
            
            df_spatial = df_geo_agg.groupby(['lat_round', 'lng_round']).agg({
                'delay_days': 'mean',
                'order_id': 'count'
            }).reset_index().rename(columns={'delay_days': 'retraso_promedio'})
            
            # Solo zonas con retrasos
            df_risk = df_spatial[df_spatial['retraso_promedio'] > 0].copy()
            
            # Crear heatmap
            from folium.plugins import HeatMap
            
            datos_calor = df_risk[['lat_round', 'lng_round', 'retraso_promedio']].values.tolist()
            
            HeatMap(
                datos_calor,
                radius=15,
                blur=20,
                max_zoom=1,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
                name='Mapa de Riesgo'
            ).add_to(mapa)
        
        hubs = results['hubs']
        
        # AÑADIR HUBS
        logger.info("   Añadiendo hubs propuestos...")
        
        for hub in hubs:
            # Marcador de Hub
            folium.Marker(
                location=[hub['lat'], hub['lng']],
                popup=folium.Popup(
                    f"<b>HUB PROPUESTO #{hub['hub_id']}</b><br>"
                    f"<b>📍 Ubicación:</b> {hub['city']}, {hub['state']}<br>"
                    f"<b>📦 Impacto:</b> {hub['load']:,} órdenes<br>"
                    f"<b>⏱️ Retraso actual:</b> {hub['avg_delay']:.1f} días<br>"
                    f"<b>📊 % del problema:</b> {hub['load']/results['n_incidents']*100:.1f}%",
                    max_width=300
                ),
                icon=folium.Icon(color='black', icon='warehouse', prefix='fa'),
                tooltip=f"🏭 Hub {hub['hub_id']}: {hub['city']}"
            ).add_to(mapa)
            
            # Círculo de cobertura
            folium.Circle(
                location=[hub['lat'], hub['lng']],
                radius=coverage_radius_km * 1000,  # km a metros
                color='black',
                weight=2,
                dash_array='5, 5',
                fill=False,
                tooltip=f"Radio de cobertura: {coverage_radius_km}km"
            ).add_to(mapa)
        
        # LEYENDA MEJORADA
        leyenda_html = f"""
        <div style="position: fixed; bottom: 30px; right: 30px; z-index:9999;
             background: rgba(255, 255, 255, 0.95); padding: 15px; 
             border: 2px solid #333; border-radius: 8px; 
             box-shadow: 4px 4px 10px rgba(0,0,0,0.2); 
             font-family: sans-serif; width: 240px;">
             
             <h4 style="margin:0 0 10px 0; color:#333; 
                        border-bottom:2px solid #333; padding-bottom:5px;">
                🏭 Expansión de Red
             </h4>
             
             <div style="margin-bottom: 10px;">
                <b style="font-size:11px; color:#555;">CONTEXTO (FONDO)</b>
                <div style="background: linear-gradient(to right, blue, lime, red); 
                            height: 8px; width: 100%; margin: 5px 0; border-radius:2px;"></div>
                <div style="font-size:10px; display:flex; justify-content:space-between;">
                    <span>Bajo riesgo</span>
                    <span>Alto riesgo</span>
                </div>
             </div>
             
             <hr style="margin: 10px 0; opacity: 0.3;">
             
             <div style="margin-bottom: 10px;">
                <b style="font-size:11px; color:#555;">HUBS PROPUESTOS</b>
                <div style="display:flex; align-items:center; margin:8px 0;">
                    <i class="fa fa-warehouse" style="color:black; font-size:18px; 
                                                      margin-right:10px;"></i>
                    <span style="font-size:11px;">Ubicación Óptima (AI)</span>
                </div>
                <div style="display:flex; align-items:center;">
                    <span style="font-size:20px; color:black; 
                                 margin-right:8px;">⭕</span>
                    <span style="font-size:11px;">Cobertura ({coverage_radius_km}km)</span>
                </div>
             </div>
             
             <hr style="margin: 10px 0; opacity: 0.3;">
             
             <div style="font-size:10px; color:#666;">
                <b>Algoritmo:</b> Weighted K-Means<br>
                <b>Total hubs:</b> {len(hubs)}<br>
                <b>Incidentes:</b> {results['n_incidents']:,}
             </div>
        </div>
        """
        
        mapa.get_root().html.add_child(folium.Element(leyenda_html))
        
        # Añadir control de capas
        folium.LayerControl(collapsed=False).add_to(mapa)
        
        logger.info(f"✅ {len(hubs)} hubs visualizados con contexto completo")
        
        return mapa
    def get_summary_report(self, results: Dict) -> str:
        """Genera reporte ejecutivo"""
        lines = ["="*70, "EXPANSIÓN DE RED LOGÍSTICA", "="*70]
        lines.append(f"Total incidentes: {results['n_incidents']:,}")
        lines.append(f"Hubs propuestos: {len(results['hubs'])}")
        lines.append("")
        
        for hub in results['hubs']:
            lines.append(f"HUB #{hub['hub_id']}: {hub['city']}, {hub['state']}")
            lines.append(f"  Coordenadas: ({hub['lat']:.2f}, {hub['lng']:.2f})")
            lines.append(f"  Impacto: {hub['load']:,} órdenes ({hub['load']/results['n_incidents']*100:.1f}%)")
            lines.append(f"  Retraso actual: {hub['avg_delay']:.1f} días")
            lines.append("")
        
        lines.append("="*70)
        return "\n".join(lines)
