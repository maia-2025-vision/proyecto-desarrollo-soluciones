import streamlit as st

st.set_page_config(page_title="Aplicación de Detección de Vacas", layout="wide", initial_sidebar_state="expanded")

st.title("Aplicación de Detección de Vacas")
st.markdown("""
Bienvenido a la Aplicación de Detección de Vacas. Elija una opción de la barra lateral:
- **Cargar Imágenes**: Suba nuevas imágenes para su procesamiento.
- **Ver Detecciones**: Visualice los resultados de las detecciones.
""")

st.sidebar.success("Seleccione una página arriba.")
