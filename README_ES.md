# 🚀 Olist Inteligencia Logística

> **[🇺🇸 English Version](README.md)**

> **[Versión en Español](README_ES.md)** | **[English Version](README.md)**

**Sistema ML end-to-end para optimización de inventarios y análisis de redes de distribución**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hecho con ❤️](https://img.shields.io/badge/Hecho%20con-❤️-red.svg)](https://github.com/magooscar86)

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características Clave](#características-clave)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Inicio Rápido](#inicio-rápido)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Tecnologías](#tecnologías)
- [Licencia](#licencia)

---

## 🎯 Descripción

Sistema integral de inteligencia logística para **Olist** (marketplace brasileño de e-commerce), procesando **110,000+ órdenes históricas** para optimizar gestión de inventarios y estrategia de red de distribución.

**Desafío de Negocio:**
- Stock de seguridad ineficiente → $31k de capital inmovilizado
- Retrasos sistemáticos en regiones desatendidas (Norte/Nordeste: 12.4 días promedio)
- Sin herramientas para cuantificar costo del riesgo geográfico

**Solución Entregada:**
- Forecasting híbrido (Gaussian Process + LightGBM) con selección automática de modelos
- Inteligencia espacial (Kriging) prediciendo retrasos en CUALQUIER coordenada
- Análisis de red revelando vulnerabilidades críticas (São Paulo: 59% dependencia)
- Optimización estratégica de hubs (96% cobertura con 3 hubs)

---

## ✨ Características Clave

### 1. 📈 Forecasting Inteligente
- **Selector híbrido**: Elige automáticamente mejor modelo por categoría (GP/LightGBM/Media Móvil)
- **Stock de seguridad dinámico**: Basado en incertidumbre predictiva (σ), no varianza histórica
- **Validación por torneo**: CV temporal 5-fold con testing de Wilcoxon
- **Impacto**: 15% reducción RMSE, $6.7k optimizados en inventario

### 2. 🗺️ Inteligencia Espacial (Kriging)
- **Gaussian Process con kernel RBF** para predicción de retrasos en 5,000+ coordenadas
- Predice riesgo de entrega **sin datos históricos** en nuevas ubicaciones
- Cuantificación geográfica de costos: "Presupuesto de incertidumbre" por región
- **Impacto**: Identificó zonas críticas requiriendo inversión en infraestructura

### 3. 🏗️ Optimización de Ubicación de Hubs
- **K-Means ponderado** priorizando "dolor logístico" (retraso²) sobre volumen de ventas
- Ubicación estratégica cubriendo **96% de zonas problemáticas**
- Filosofía: *Ubicar donde fallamos, no donde vendemos*
- **Impacto**: Reducción proyectada 45% en retrasos con 3 hubs

### 4. 🕸️ Análisis de Vulnerabilidad de Red
- **Grafo dirigido**: 4,000 nodos (ciudades), 5,000 aristas (rutas)
- **Métricas de centralidad**: Grado, Intermediación (NetworkX)
- Identifica puentes críticos (Cotia: 2.3%) y monocentrismo
- **Impacto**: Cuantificó riesgo sistémico (concentración en SP)

### 5. 🎨 Dashboards Interactivos
- Calculadora de inventario en tiempo real (categoría + nivel servicio)
- Evaluación de riesgo geográfico (cualquier coordenada)
- Reportes ejecutivos de 4 paneles para tomadores de decisión

---

### 🎮 Demostraciones en Vivo

#### 🖥️ Olist AI Command Center
Panel interactivo para toma de decisiones estratégicas con optimización de inventario en tiempo real, cálculos de nivel de servicio y predicción de retrasos por categoría.

![Olist AI Command Center](outputs/Olist%20AI%20Command%20Center.png)

#### 📦 Calculadora de Envíos
Herramienta de evaluación de riesgo geográfico que utiliza modelos espaciales Kriging para predecir retrasos de entrega en cualquier coordenada de Brasil.

![Calculadora de Envíos](outputs/Calculadora%20de%20envios.png)

---

## 📂 Estructura del Proyecto
```
Olist_Project/
│
├── src/                          # Código fuente (modular)
│   ├── data/                     # Carga de datos e ingeniería de features
│   ├── models/                   # Modelos ML (GP, LightGBM, baselines)
│   ├── visualization/            # Dashboards interactivos
│   ├── optimization/             # Algoritmos de ubicación de instalaciones
│   ├── network/                  # Análisis de grafos
│   ├── apps/                     # Aplicaciones para usuarios
│   └── api/                      # API REST (FastAPI)
│
├── data/raw/                     # Datasets originales (Kaggle Olist)
├── outputs/                      # Artefactos generados (mapas, reportes)
├── notebooks/                    # Flujo de análisis
│   └── Olist_Executive_Report.ipynb
│
├── Dockerfile                    # Definición de contenedor
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

---

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8+
- 4GB RAM (para modelo espacial GP)
- ~500MB espacio en disco

### Configuración Local

\`\`\`bash
# Clonar repositorio
git clone https://github.com/magooscar86/olist-logistics-intelligence.git
cd olist-logistics-intelligence

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
\`\`\`

### Docker (Alternativa)

\`\`\`bash
docker-compose up -d
# API: http://localhost:8000
# Jupyter: http://localhost:8888
\`\`\`

---

## 🚀 Inicio Rápido

### 1. Descargar Datos
\`\`\`bash
# Obtener dataset Olist de Kaggle:
# https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Colocar CSVs en data/raw/
\`\`\`

### 2. Ejecutar Análisis
\`\`\`python
from src.data.data_loader import OlistDataLoader
from src.models.model_selector import ModelSelector

loader = OlistDataLoader()
df_main = loader.load_and_merge()

selector = ModelSelector(df_main)
results = selector.run_tournament()
selector.visualize_results()
\`\`\`

### 3. Dashboard Interactivo
\`\`\`python
from src.visualization.interactive_ui import OlistDashboard

dashboard = OlistDashboard(df_main)
dashboard.render()
\`\`\`

---

## 🐳 Avanzado: Deployment de API (Opcional)

> **Nota:** Docker NO es necesario para ejecutar el notebook de análisis. Solo se requiere si quieres desplegar los modelos como API REST.

### ¿Qué es la API?

El proyecto incluye un servidor FastAPI listo para producción que expone los modelos entrenados como endpoints HTTP:

**Endpoints:**
- `GET /health` - Health check
- `POST /predict/shipping` - Predecir retrasos de entrega por coordenadas
- `POST /predict/forecast` - Forecast de demanda por categoría
- `POST /optimize/inventory` - Calcular stock de seguridad óptimo
- `POST /analyze/network` - Análisis de vulnerabilidad de red

### Inicio Rápido (Docker)
```bash
# Iniciar servidor API
docker-compose up -d

# Acceder a documentación Swagger
open http://localhost:8000/docs

# Probar endpoint
curl -X POST http://localhost:8000/predict/shipping \
  -H "Content-Type: application/json" \
  -d '{"latitude": -23.55, "longitude": -46.63}'
```

**Requisitos:**
- Docker Desktop instalado
- Modelos entrenados en `checkpoints/` (generados al ejecutar notebook primero)

**Guía Docker detallada:** Ver [docs/DOCKER_README.md](docs/DOCKER_README.md)

---

## 🎨 Panorama Visual

### 🗺️ Inteligencia Espacial Interactiva

Explora las predicciones de retrasos de entrega en Brasil con nuestro modelo Gaussian Process Kriging:

#### Mapa Maestro de Predicciones
[![Mapa de Calor Espacial](docs/images/mapa_maestro_predicciones.png)](https://magooscar86.github.io/olist-logistics-intelligence/maps/mapa_maestro_predicciones.html)

**[🔗 Abrir Mapa Interactivo](https://magooscar86.github.io/olist-logistics-intelligence/maps/mapa_maestro_predicciones.html)** - Haz clic para explorar predicciones en vivo

---

#### Ubicaciones Óptimas de Hubs
[![Optimización de Hubs](docs/images/mapa_hubs.png)](https://magooscar86.github.io/olist-logistics-intelligence/maps/mapa_hubs.html)

**[🔗 Ver Análisis de Hubs](https://magooscar86.github.io/olist-logistics-intelligence/maps/mapa_hubs.html)** - Optimización de ubicación de instalaciones con K-Means

---

#### Rendimiento Histórico
[![Análisis Histórico](docs/images/mapa_historico.png)](https://magooscar86.github.io/olist-logistics-intelligence/maps/mapa_historico.html)

**[🔗 Explorar Datos Históricos](https://magooscar86.github.io/olist-logistics-intelligence/maps/mapa_historico.html)** - Mapa de calor de rendimiento de entregas

---

### 📊 Dashboards Analíticos

<table>
<tr>
<td width="50%">

#### Resultados del Torneo de Modelos
![Dashboard Torneo](docs/images/dashboard_torneo.png)
*Análisis comparativo de algoritmos de forecasting (GP, LightGBM, Baselines)*

</td>
<td width="50%">

#### Reporte de Impacto Financiero
![Reporte Financiero](docs/images/reporte_financiero_v2.png)
*Ahorros de costos y métricas de optimización*

</td>
</tr>
<tr>
<td width="50%">

#### Optimización de Inventario
![Análisis de Stock](docs/images/stock_analysis.png)
*Cálculos de stock de seguridad por categoría*

</td>
<td width="50%">

#### Mapa de Calor Comparación de Modelos
![Comparación de Modelos](docs/images/heatmap_comparacion.png)
*Métricas de rendimiento de validación cruzada*

</td>
</tr>
</table>

---

---

## 🔬 Metodología

### Pipeline de Forecasting
1. Preparación: 109k órdenes → Agregación semanal por categoría
2. Ingeniería de features: Lags, estadísticas móviles, descomposición estacional
3. Torneo de modelos: GP vs LightGBM vs Media Móvil (CV)
4. Stock de seguridad: `SS = Z-score × σ × √(Lead Time)`

### Inteligencia Espacial
1. Kriging con kernel RBF (length_scale ~329km aprendido)
2. Entrenamiento: 4,000 coordenadas históricas
3. Inferencia: Predicción en CUALQUIER coordenada (interpolación)

### Análisis de Red
1. DiGraph: ciudad_vendedor → ciudad_cliente (ponderado por volumen)
2. Métricas: Out-degree, In-degree, Betweenness centrality
3. Vulnerabilidad: Índices de concentración, simulación de cascadas

---

## 📊 Resultados

| Métrica | Baseline | Este Sistema | Mejora |
|---------|----------|--------------|--------|
| RMSE Forecast | 47.2 | 40.1 | ✅ -15% |
| Capital inmovilizado | $38k | $31.3k | ✅ -$6.7k (18%) |
| Cobertura problema | - | 96% | ✅ Con 3 hubs |
| Reducción retrasos | - | -45% | ✅ Proyectado |

**Insights Clave:**
- 🔴 Monocentrismo: São Paulo = 59.3% conectividad de red
- 🟡 Ineficiencia Nordeste: 12.4 días promedio de retraso
- 🟢 Especialización: Ibitinga (textil) = 30.4% a pesar de tamaño pequeño

---

## 🛠️ Tecnologías

**Core:** Python, pandas, numpy, scikit-learn  
**ML:** Gaussian Processes, LightGBM, NetworkX  
**Geo:** folium, Kriging (kernel RBF)  
**Viz:** matplotlib, seaborn, ipywidgets  
**Infra:** FastAPI, Docker  

---

## 📜 Licencia

Licencia MIT - Ver [LICENSE](LICENSE)

---

## 👤 Autor

**Oscar Antonio Melo Leon**  
📧 Email: magooscar86@gmail.com  
💼 LinkedIn: [linkedin.com/in/oscar-antonio-león-36b73a105](https://www.linkedin.com/in/oscar-antonio-león-36b73a105)  
🐙 GitHub: [magooscar86](https://github.com/magooscar86)

---

## 🙏 Agradecimientos

- Dataset: [Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Inspiración: Amazon DeepAR, UPS ORION
- Comunidad: scikit-learn, NetworkX

---

**⭐ ¡Dale estrella al repo si te fue útil!**
