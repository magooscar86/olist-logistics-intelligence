# Contributing to Olist Logistics Intelligence

¡Gracias por tu interés en contribuir! 🎉

Este documento te ayudará a comenzar a contribuir al proyecto.

---

## 🚀 Maneras de Contribuir

- **Reportar bugs** - Ayúdanos a encontrar y solucionar problemas
- **Sugerir features** - Ideas nuevas para mejorar el proyecto
- **Mejorar documentación** - Corrige errores o agrega ejemplos
- **Código** - Implementa nuevas funcionalidades o mejoras
- **Revisiones** - Revisa pull requests de otros contribuidores

---

## 📋 Requisitos Previos

Antes de contribuir, asegúrate de tener:

1. **Python 3.8+** instalado
2. **Git** configurado
3. **Conocimiento básico** de:
   - Machine Learning (forecasting, spatial analysis)
   - Python y librerías del proyecto
   - Jupyter Notebooks

---

## 🛠️ Configuración del Entorno de Desarrollo

```bash
# 1. Clona el repositorio
git clone https://github.com/magooscar86/olist-logistics-intelligence.git
cd olist-logistics-intelligence

# 2. Crea un entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 3. Instala el proyecto en modo desarrollo
pip install -e ".[dev]"

# 4. Descarga el dataset de Kaggle
# https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Coloca los archivos CSV en: data/raw/
```

---

## 📝 Estilo de Código

Seguimos las mejores prácticas de Python:

| Herramienta | Configuración |
|-------------|----------------|
| **Black** | line-length = 100 |
| **Ruff** | Seleccionados: E, F, I, N, W |
| **MyPy** | Python 3.8+, warn_return_any = true |

### Antes de hacer commit:

```bash
# Verifica el código
black --check src/ notebooks/
ruff check src/ notebooks/
mypy src/ --ignore-missing-imports
```

---

## 🔄 Proceso de Contribución

### 1. Fork el repositorio

Haz clic en el botón "Fork" en GitHub.

### 2. Crea una rama

```bash
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/corregir-bug
```

### 3. Haz tus cambios

- Sigue las convenciones de código
- Agrega docstrings a las funciones
- Escribe comentarios solo cuando sea necesario

### 4. Commits significativos

```bash
git add .
git commit -m "feat: agregar nuevo modelo de forecasting"
# Tipos: feat, fix, docs, style, refactor, test, chore
```

### 5. Push y Pull Request

```bash
git push origin feature/nueva-funcionalidad
```

Luego, crea un Pull Request en GitHub.

---

## 📖 Estructura del Proyecto

```
olist-logistics-intelligence/
├── src/
│   ├── api/              # API REST con FastAPI
│   ├── data/             # Carga y procesamiento de datos
│   ├── models/           # Modelos ML (GP, LightGBM)
│   ├── network/          # Análisis de grafos
│   ├── optimization/     # Optimización de ubicaciones
│   ├── utils/            # Utilidades
│   └── visualization/    # Visualizaciones
├── notebooks/            # Jupyter Notebooks
├── docs/                # Documentación y mapas
├── data/raw/            # Dataset Olist (descargar de Kaggle)
└── outputs/             # Resultados generados
```

---

## ❓ Preguntas?

- Abre un **Issue** para discutir ideas
- Únete a nuestras discusiones
- Consulta la documentación en el wiki

---

## 📜 Código de Conducta

Al participar, debes respetar el [Código de Conducta](https://github.com/magooscar86/olist-logistics-intelligence/blob/main/CODE_OF_CONDUCT.md).

---

¡期待你的贡献! (¡esperamos tu contribución!)
