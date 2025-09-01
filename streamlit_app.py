import streamlit as st

st.set_page_config(
    page_title="Aplicación de Detección de Vacas",
    layout="wide"
)

st.title("Aplicación de Detección de Vacas")
st.markdown("""
Bienvenido a la Aplicación de Detección de Vacas. Elija una opción de la barra lateral:

- **Cargar Imágenes**: Cargar imágenes al bucket S3 para procesamiento
- **Ver Detecciones**: Mostrar imágenes con cuadros delimitadores detectados
""")

st.sidebar.success("Seleccione una página arriba.")