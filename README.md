# Clasificador de Alimentos y Nutrición

Proyecto desarrollado para el curso de Inteligencia Artificial de la Universidad Mariano Gálvez de Guatemala.

El sistema implementa técnicas de Deep Learning y Visión por Computadora para identificar alimentos mediante imágenes y clasificarlos según su categoría nutricional de forma educativa.

---

# Descripción del Proyecto

El proyecto consiste en una aplicación web desarrollada en Python que permite detectar alimentos o bebidas a partir de fotografías utilizando un modelo de Inteligencia Artificial entrenado con redes neuronales convolucionales.

La aplicación clasifica los alimentos en categorías nutricionales como:

- Fresco/Saludable
- Procesado
- Ultraprocesado
- Bebida azucarada

Además, el sistema muestra:

- Nombre del alimento detectado
- Nivel de confianza de la predicción
- Categoría nutricional
- Recomendaciones educativas

El objetivo principal es proporcionar una herramienta educativa que ayude a generar mayor conciencia sobre los hábitos alimenticios sin realizar diagnósticos médicos ni recomendaciones clínicas.

---

# Objetivos

## Objetivo General

Desarrollar una aplicación web basada en Inteligencia Artificial capaz de clasificar alimentos mediante imágenes utilizando Deep Learning.

## Objetivos Específicos

- Implementar un modelo de clasificación de alimentos en tiempo real y carga de imagenes.
- Utilizar Transfer Learning con EfficientNet-B0.
- Mostrar información nutricional básica de forma visual.
- Permitir interacción mediante una interfaz web accesible.
- Aplicar buenas prácticas de entrenamiento y evaluación de modelos IA.

---

# Tecnologías Principales Utilizadas

| Tecnología | Uso |
|---|---|
| Python | Lenguaje principal |
| PyTorch | Deep Learning |
| Torchvision | Modelos preentrenados |
| Streamlit | Interfaz web |
| OpenCV | Procesamiento multimedia |
| NumPy | Operaciones matemáticas |
| Scikit-learn | Métricas |
| Matplotlib | Visualización |
| GitHub | Control de versiones |

---

# Modelo de Inteligencia Artificial

El sistema utiliza el modelo:

- EfficientNet-B0

Características principales:

- Transfer Learning
- Fine-Tuning
- Clasificación multiclase
- Predicción mediante Softmax
- Compatibilidad CPU/GPU

---

# Dataset Utilizado

El modelo fue entrenado utilizando:

- Food-101 (12 clases)
- Fruits-360 (5 clases)
- Imágenes locales (4 clases)

---

# Características Principales

- Clasificación de alimentos mediante imágenes y camara en tiempo real
- Interfaz web intuitiva
- Compatible con CPU y GPU
- Soporte para imágenes locales
- Clasificación nutricional educativa
- Uso de Deep Learning moderno

---

# Estructura General del Proyecto

```text
CLASIFICADOR DE ALIMENTOS/
│
├── data/
│   └── food-101/
│       ├── images/
│       ├── meta/
│       ├── license_agreement.txt
│       └── README.txt
│
├── docs/
│   ├── Manual Técnico.pdf
│   ├── Manual De Usuario.pdf
│   ├── Informe Final.pdf
│   └── DERCAS.pdf
│
├── models/
│   ├── classes.json
│   ├── food_info.json
│   ├── modelo_efficientnet_V1.pth
│   └── modelo_efficientnet.pth
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_train.ipynb
│   └── 03_evaluation.ipynb
│
├── venv310/
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```
# Documentación del Proyecto

La documentación completa se encuentra en la carpeta `docs/`.

| Documento | Descripción |
|---|---|
| [ManualTecnico.pdf](docs/Manual Tecnico.pdf) | Instalación, configuración y arquitectura del sistema |
| [ManualUsuario.pdf](docs/manual_de_usuario.pdf) | Guía de uso de la aplicación |
| [InformeFinal.pdf](docs/Informe Final.pdf) | Desarrollo, métricas y resultados del proyecto |
| [DERCAS.pdf](docs/DERCAS.pdf) | Análisis y diseño del sistema |

---

# Repositorio y Control de Versiones

El proyecto utiliza GitHub como sistema de control de versiones y trazabilidad de cambios mediante commits realizados por cada integrante del equipo.

---

# Consideraciones Éticas

- El sistema tiene únicamente fines educativos.
- No proporciona diagnósticos médicos.
- No calcula calorías exactas.
- No sustituye asesoría nutricional profesional.

---

# Posibles Mejoras Futuras

- Integración con APIs nutricionales
- Soporte multilenguaje
- Historial de predicciones
- Mayor cantidad de clases
- Detección de empaques
- Despliegue en la nube

---

# Equipo de Desarrollo

Proyecto realizado por estudiantes de la Universidad Mariano Gálvez de Guatemala para el curso de Inteligencia Artificial.

---

# Licencia

Proyecto académico desarrollado con fines educativos.