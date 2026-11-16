"""CañaSana IA: diagnóstico de enfermedades en hojas de caña de azúcar con redes neuronales.

Aplicación Streamlit que usa el mejor modelo del proyecto (EfficientNetB0 con ajuste fino).
Ejecutar en local:  streamlit run app.py
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parent
TAM = 224
COLORES = {
    "Sana": "#2e7d32",
    "Mosaico": "#7b6c14",
    "Pudrición roja": "#a83b46",
    "Roya": "#b76022",
    "Hoja amarilla": "#c9a227",
}

st.set_page_config(page_title="CañaSana IA", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {max-width: 1200px;}
    .cabecera {background: linear-gradient(115deg, #1f4d2c, #2e7d32); border-radius: 14px;
               padding: 22px 28px; margin-bottom: 14px;}
    .cabecera h1 {color: #fff; margin: 0 0 4px 0; font-size: 1.8rem;}
    .cabecera p {color: #d9f0dc; margin: 0;}
    .chips {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;}
    .chips span {background: #e6f2e7; color: #1f4d2c; padding: 3px 10px; border-radius: 999px; font-size: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo de redes neuronales…")
def cargar_modelo():
    import keras

    modelo = keras.models.load_model(RAIZ / "modelo" / "modelo_cana.keras", compile=False)
    info = json.loads((RAIZ / "modelo" / "clases.json").read_text(encoding="utf-8"))
    metricas = json.loads((RAIZ / "modelo" / "metricas.json").read_text(encoding="utf-8"))
    return modelo, info, metricas


def preparar(imagen):
    """Mismo preprocesamiento que en el entrenamiento: RGB, orientación EXIF y 224 x 224 píxeles."""
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    return np.asarray(imagen.resize((TAM, TAM), Image.BILINEAR), dtype=np.float32)


def diagnosticar(imagenes):
    """Recibe una lista de imágenes PIL y devuelve una lista de diagnósticos."""
    modelo, info, _ = cargar_modelo()
    lote = np.stack([preparar(im) for im in imagenes])
    inicio = time.perf_counter()
    probabilidades = np.asarray(modelo(lote, training=False))
    ms = (time.perf_counter() - inicio) * 1000 / len(imagenes)
    resultados = []
    for prob in probabilidades:
        k = int(prob.argmax())
        clase = info["clases"][k]
        resultados.append({
            "clase": clase,
            "nombre": info["nombres_es"][clase],
            "confianza": float(prob[k]),
            "probabilidades": {info["nombres_es"][c]: float(p) for c, p in zip(info["clases"], prob)},
            "recomendacion": info["recomendaciones"][clase],
            "consultar": float(prob[k]) < info["umbral_confianza_especialista"],
            "ms": ms,
        })
    return resultados


def grafico_probabilidades(probabilidades):
    tabla = pd.DataFrame({"Clase": list(probabilidades), "Probabilidad": list(probabilidades.values())})
    return (
        alt.Chart(tabla)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Clase:N", sort="-x", title=None),
            x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            color=alt.Color("Clase:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values())),
                            legend=None),
            tooltip=["Clase:N", alt.Tooltip("Probabilidad:Q", format=".1%")],
        )
        .properties(height=210)
    )


def mostrar_resultado(imagen, resultado):
    col_img, col_res = st.columns([1, 1.3])
    with col_img:
        st.image(imagen, caption="Imagen analizada", width="stretch")
    with col_res:
        texto = f"### {resultado['nombre']}\nConfianza del modelo: **{resultado['confianza']:.1%}**"
        if resultado["clase"] == "Healthy":
            st.success(texto)
        else:
            st.error(texto)
        if resultado["consultar"]:
            st.warning("La confianza es menor al 70 %: el resultado es dudoso. Consulte a un especialista.")
        st.markdown(f"**Recomendación:** {resultado['recomendacion']}")
        st.altair_chart(grafico_probabilidades(resultado["probabilidades"]), width="stretch")
        st.caption(f"Tiempo de inferencia: {resultado['ms']:.0f} ms")


# ----------------------------------------------------------------------------- interfaz
modelo, info, metricas = cargar_modelo()

st.markdown(
    """
    <section class="cabecera">
      <h1>🌿 CañaSana IA</h1>
      <p>Detección de enfermedades en hojas de caña de azúcar mediante redes neuronales · Región La Libertad</p>
      <div class="chips"><span>EfficientNetB0 + ajuste fino</span><span>5 clases</span>
      <span>Aprendizaje por transferencia</span><span>Python · Streamlit</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Modelo")
    st.metric("Exactitud en prueba", f"{metricas['exactitud']:.1%}")
    st.metric("F1-score macro", f"{metricas['f1_macro']:.3f}")
    st.caption(f"Evaluado con {metricas['imagenes_prueba']} imágenes de prueba "
               f"de un total de {metricas['imagenes_total']} imágenes limpias.")
    st.divider()
    st.markdown("**Clases que reconoce**")
    for nombre in info["nombres_es"].values():
        st.markdown(f"- {nombre}")
    st.divider()
    st.caption("Herramienta de apoyo: no reemplaza el diagnóstico de un fitopatólogo.")

pestana_1, pestana_2, pestana_3 = st.tabs(["01 · Diagnóstico de una hoja", "02 · Análisis por lotes",
                                            "03 · Acerca del modelo"])

with pestana_1:
    st.subheader("Analizar una fotografía")
    origen = st.radio("¿De dónde viene la imagen?", ["Subir una foto", "Usar la cámara", "Usar una imagen de ejemplo"],
                      horizontal=True)
    imagen = None
    if origen == "Subir una foto":
        archivo = st.file_uploader("Selecciona una foto de la hoja (JPG o PNG)", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            imagen = Image.open(archivo)
    elif origen == "Usar la cámara":
        foto = st.camera_input("Toma una foto de la hoja con buena luz")
        if foto is not None:
            imagen = Image.open(foto)
    else:
        ejemplos = sorted((RAIZ / "ejemplos").glob("*.jpg"))
        nombres = {p.stem.replace("_", " ").capitalize(): p for p in ejemplos}
        elegido = st.selectbox("Imagen de ejemplo (conjunto de prueba)", list(nombres))
        if st.button("Diagnosticar ejemplo", type="primary"):
            imagen = Image.open(nombres[elegido])

    if imagen is not None:
        with st.spinner("Analizando la hoja…"):
            resultado = diagnosticar([imagen])[0]
        mostrar_resultado(imagen, resultado)

with pestana_2:
    st.subheader("Analizar varias fotos a la vez")
    archivos = st.file_uploader("Selecciona varias fotos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                key="lote")
    if archivos and st.button("Analizar lote", type="primary"):
        with st.spinner(f"Analizando {len(archivos)} imágenes…"):
            imagenes = [Image.open(a) for a in archivos]
            resultados = diagnosticar(imagenes)
        tabla = pd.DataFrame({
            "Archivo": [a.name for a in archivos],
            "Diagnóstico": [r["nombre"] for r in resultados],
            "Confianza": [r["confianza"] for r in resultados],
            "Consultar especialista": ["Sí" if r["consultar"] else "No" for r in resultados],
        })
        enfermas = int((tabla["Diagnóstico"] != "Sana").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Hojas analizadas", len(tabla))
        c2.metric("Con signos de enfermedad", enfermas)
        c3.metric("Porcentaje afectado", f"{enfermas / len(tabla):.0%}")
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={"Confianza": st.column_config.ProgressColumn(
                         "Confianza", min_value=0.0, max_value=1.0, format="percent")})
        conteo = tabla["Diagnóstico"].value_counts().rename_axis("Clase").reset_index(name="Hojas")
        st.altair_chart(
            alt.Chart(conteo).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Clase:N", sort="-x", title=None), x=alt.X("Hojas:Q", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Clase:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values())),
                                legend=None)),
            width="stretch")
        st.download_button("Descargar reporte CSV", tabla.to_csv(index=False).encode("utf-8-sig"),
                           file_name="reporte_cana.csv", mime="text/csv")

with pestana_3:
    st.subheader("¿Cómo funciona?")
    st.markdown(
        "1. La foto se redimensiona a 224 × 224 píxeles.\n"
        "2. Una red neuronal convolucional **EfficientNetB0**, preentrenada con ImageNet y ajustada con fotos de "
        "hojas de caña (aprendizaje por transferencia), calcula la probabilidad de cada clase.\n"
        "3. La app muestra la clase más probable, su confianza y una recomendación; si la confianza es menor al "
        "70 %, sugiere consultar a un especialista."
    )
    st.subheader("Desempeño en el conjunto de prueba")
    st.dataframe(
        pd.DataFrame({"Clase": list(metricas["recall_por_clase"]),
                      "Sensibilidad (recall)": list(metricas["recall_por_clase"].values())}),
        hide_index=True, width="stretch",
        column_config={"Sensibilidad (recall)": st.column_config.ProgressColumn(
            "Sensibilidad (recall)", min_value=0.0, max_value=1.0, format="percent")},
    )
    st.subheader("Datos y limitaciones")
    st.markdown(
        "- Dataset: *Sugarcane Leaf Disease Dataset* (Daphal y Koli, 2022), Mendeley Data, licencia CC BY 4.0, "
        "https://doi.org/10.17632/9424skmnrk.1.\n"
        "- Las fotos de entrenamiento se tomaron en la India; con fotos de La Libertad el desempeño puede variar.\n"
        "- Solo reconoce las 5 clases indicadas; otras enfermedades o plagas pueden clasificarse de forma errónea."
    )
