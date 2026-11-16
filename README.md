# CañaSana IA 🌿

Aplicación web que detecta enfermedades en hojas de caña de azúcar a partir de una fotografía, usando una red
neuronal convolucional **EfficientNetB0 con ajuste fino** (aprendizaje por transferencia).

Proyecto del curso *Sistemas Inteligentes y Machine Learning*, Universidad Privada del Norte (UPN), 2026-2.

## ¿Qué hace?
- **Diagnóstico de una hoja:** subes una foto, usas la cámara o eliges una imagen de ejemplo. La app muestra la clase
  más probable, la confianza, las probabilidades de cada clase y una recomendación.
- **Análisis por lotes:** analiza varias fotos a la vez y permite descargar un reporte CSV.
- **Acerca del modelo:** explica cómo funciona y su desempeño.

Clases: hoja sana, mosaico, pudrición roja, roya y hoja amarilla.
Desempeño en el conjunto de prueba (345 imágenes): **exactitud 89.0 %**, **F1-score macro 0.889**.

## Estructura
```
app.py                 aplicación Streamlit
requirements.txt       librerías necesarias
modelo/                modelo entrenado (modelo_cana.keras), clases y métricas
ejemplos/              una imagen de ejemplo por clase (conjunto de prueba)
.streamlit/config.toml colores de la app
```

## Ejecutar en una computadora
1. Abrir **Anaconda Prompt** y activar el entorno: `conda activate proyecto`
2. Ir a esta carpeta: `cd "ruta\a\app_streamlit"`
3. Ejecutar: `streamlit run app.py` (se abre el navegador en http://localhost:8501)

## Datos
Daphal, S., & Koli, S. (2022). *Sugarcane Leaf Disease Dataset* (Versión 1) [Conjunto de datos]. Mendeley Data.
https://doi.org/10.17632/9424skmnrk.1 — licencia CC BY 4.0. Las imágenes de `ejemplos/` provienen de este dataset.

## Aviso
Es una herramienta de apoyo: no reemplaza el diagnóstico de un fitopatólogo. Las fotos de entrenamiento se tomaron en la
India; con fotografías de otros lugares el desempeño puede variar.
