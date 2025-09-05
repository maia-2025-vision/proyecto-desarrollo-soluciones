import streamlit as st

st.set_page_config(
    page_title="Detección de Ganado 🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Detección de Ganado 🐄")
st.markdown("""
### Navegación
Use la barra lateral para navegar entre las páginas de la aplicación.
- **Cargar Imágenes:** Para subir nuevas imágenes y procesarlas.
- **Mostrar Detecciones:** Para mostrar los resultados de las imágenes procesadas.
""")

# Definir titulos para las paginas
st.sidebar.page_link("pages/1_Upload_Images.py", label="Cargar Imágenes", icon="📤")
st.sidebar.page_link("pages/2_View_Detections.py", label="Mostrar Detecciones", icon="🖼️")
