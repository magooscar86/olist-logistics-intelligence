"""
Calculadora Inteligente de Factibilidad de Envíos - Olist Logistics
"""

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from shapely.geometry import Point
from shapely.prepared import prep
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShippingCalculator:
    """Calculadora interactiva de factibilidad de envíos"""
    
    def __init__(self, gp_model, df_main: pd.DataFrame):
        self.gp_model = gp_model
        self.df_main = df_main
        self.thresh_high = None
        self.thresh_low = None
        self.prep_brazil = None
        self.df_spatial_cities = None
        
        logger.info("ShippingCalculator inicializado")
        
        self._calibrate_thresholds()
        self._prepare_geocoding()
        self._load_brazil_geometry()
    
    def _calibrate_thresholds(self):
        """Calibra umbrales de confianza"""
        logger.info("Calibrando umbrales...")
        
        df_geo = self.df_main.copy()
        df_geo['lat_round'] = df_geo['geolocation_lat'].round(2)
        df_geo['lng_round'] = df_geo['geolocation_lng'].round(2)
        
        df_spatial = df_geo.groupby(['lat_round', 'lng_round']).agg({'delay_days': 'mean'}).reset_index()
        
        sample_size = min(1000, len(df_spatial))
        df_sample = df_spatial.sample(sample_size, random_state=42)
        
        X_sample = df_sample[['lat_round', 'lng_round']].values
        _, sigma_calib = self.gp_model.predict(X_sample, return_std=True)
        
        self.thresh_high = np.percentile(sigma_calib, 50)
        self.thresh_low = np.percentile(sigma_calib, 90)
        
        logger.info(f"   Alta confianza: σ < {self.thresh_high:.2f}")
        logger.info(f"   Baja confianza: σ > {self.thresh_low:.2f}")
    
    def _prepare_geocoding(self):
        """Prepara índice de ciudades"""
        logger.info("Preparando geocoding...")
        
        self.df_spatial_cities = self.df_main.groupby(['geolocation_lat', 'geolocation_lng']).agg({
            'delay_days': 'mean',
            'customer_city': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
        }).reset_index()
        
        logger.info(f"   Índice: {len(self.df_spatial_cities):,} ubicaciones")
    
    def _load_brazil_geometry(self):
        """Carga geometría de Brasil"""
        logger.info("Cargando geometría...")
        
        url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
        gdf_brazil = gpd.read_file(url)[['geometry']]
        
        try:
            pais_brazil = gdf_brazil.union_all()
        except AttributeError:
            pais_brazil = gdf_brazil.unary_union
        
        self.prep_brazil = prep(pais_brazil)
        logger.info("   ✅ Geometría cargada")
    
    def get_nearest_city(self, lat: float, lng: float) -> str:
        """Encuentra ciudad más cercana"""
        distancias = (self.df_spatial_cities['geolocation_lat'] - lat)**2 +                     (self.df_spatial_cities['geolocation_lng'] - lng)**2
        
        idx_min = distancias.idxmin()
        ciudad = self.df_spatial_cities.loc[idx_min, 'customer_city']
        
        return ciudad.title()
    
    def predict_shipping(self, lat: float, lng: float) -> dict:
        """Predice factibilidad de envío"""
        punto = Point(lng, lat)
        if not self.prep_brazil.contains(punto):
            return {'valid': False, 'error': 'Coordenadas fuera de Brasil'}
        
        X_new = np.array([[lat, lng]])
        y_pred, sigma = self.gp_model.predict(X_new, return_std=True)
        
        val_pred = y_pred[0]
        incertidumbre = sigma[0]
        ciudad = self.get_nearest_city(lat, lng)
        
        if val_pred < 0:
            estado = "ANTICIPADA"
            color_bg = "#d4edda"
            color_text = "#155724"
            icono = "🚀"
            margen = abs(val_pred)
            recomendacion = f"Es seguro prometer entrega rápida. Holgura: {margen:.1f} días."
        else:
            estado = "RETRASO"
            color_bg = "#f8d7da"
            color_text = "#721c24"
            icono = "⚠️"
            margen = val_pred
            recomendacion = f"<b>ACCIÓN:</b> Añadir {int(margen)+1} días al ETA."
        
        if incertidumbre <= self.thresh_high:
            confianza = "ALTA"
            conf_color = "green"
            conf_msg = "Muchos datos históricos."
        elif incertidumbre <= self.thresh_low:
            confianza = "MEDIA"
            conf_color = "orange"
            conf_msg = "Inferencia regional."
        else:
            confianza = "BAJA"
            conf_color = "red"
            conf_msg = "Zona con poca información."
        
        return {
            'valid': True, 'lat': lat, 'lng': lng, 'ciudad': ciudad,
            'prediccion': val_pred, 'margen': margen,
            'incertidumbre': incertidumbre, 'estado': estado,
            'icono': icono, 'color_bg': color_bg, 'color_text': color_text,
            'recomendacion': recomendacion, 'confianza': confianza,
            'conf_color': conf_color, 'conf_msg': conf_msg
        }
    
    def render(self):
        """Renderiza interfaz"""
        logger.info("Renderizando interfaz...")
        
        w_lat = widgets.FloatText(description='🌐 Latitud:', value=-23.55, step=0.01, 
                                 layout=widgets.Layout(width='45%'))
        w_lng = widgets.FloatText(description='🌐 Longitud:', value=-46.63, step=0.01,
                                 layout=widgets.Layout(width='45%'))
        
        btn_calc = widgets.Button(description=' Analizar Envío', button_style='primary',
                                  icon='map-marker', layout=widgets.Layout(width='92%', margin='15px 0'))
        
        out_calc = widgets.Output()
        
        def on_click(b):
            with out_calc:
                clear_output()
                result = self.predict_shipping(w_lat.value, w_lng.value)
                
                if not result['valid']:
                    display(widgets.HTML(
                        f"<div style='background:#f8d7da; color:#721c24; padding:10px; border-radius:5px;'>"
                        f"❌ {result['error']}</div>"
                    ))
                    return
                
                html = f"""
                <div style="font-family: 'Segoe UI', sans-serif; border: 1px solid #ccc; 
                            border-radius: 10px; overflow: hidden; width: 450px; 
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <div style="background-color: #2c3e50; color: white; padding: 15px; 
                                display:flex; justify-content:space-between;">
                        <div>
                            <h3 style="margin:0; font-size:18px;">📦 Análisis Logística</h3>
                            <span style="font-size:12px; opacity:0.8;">{result['lat']:.2f}, {result['lng']:.2f}</span>
                        </div>
                        <div>
                            <span style="font-size:14px; font-weight:bold; 
                                         background:rgba(255,255,255,0.2); padding:3px 8px; 
                                         border-radius:4px;">📍 {result['ciudad']}</span>
                        </div>
                    </div>
                    <div style="padding: 25px; background-color: {result['color_bg']}; 
                                color: {result['color_text']}; text-align: center;">
                        <span style="font-size: 12px; font-weight: bold; 
                                     text-transform:uppercase;">PREDICCIÓN</span><br>
                        <div style="font-size: 42px; font-weight: 900; margin: 5px 0;">
                            {result['icono']} {result['margen']:.1f} <small style="font-size:16px">días</small>
                        </div>
                        <span style="font-size: 16px; font-weight:600;">{result['estado']}</span>
                    </div>
                    <div style="padding: 15px; background-color: white;">
                        <div style="display: flex; justify-content: space-between; 
                                    border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px;">
                            <span style="color: #666; font-size:13px;">Certeza:</span>
                            <div style="text-align:right;">
                                <b style="color:{result['conf_color']}">{result['confianza']}</b><br>
                                <span style="font-size:10px; color:#999;">(σ: {result['incertidumbre']:.2f})</span>
                            </div>
                        </div>
                        <p style="font-size: 11px; color: #555; margin:0;">
                            <i>{result['conf_msg']}</i>
                        </p>
                    </div>
                    <div style="background-color: #f1f2f6; padding: 12px; font-size: 12px; 
                                color: #333; border-top: 1px solid #ddd;">
                        💡 <b>Recomendación:</b> {result['recomendacion']}
                    </div>
                </div>
                """
                display(widgets.HTML(html))
        
        btn_calc.on_click(on_click)
        
        ui = widgets.VBox([
            widgets.HTML("<h3>📍 Calculadora de Envíos</h3>"),
            widgets.HBox([w_lat, w_lng]),
            btn_calc,
            out_calc
        ])
        
        display(ui)
