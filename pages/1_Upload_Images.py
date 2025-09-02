import os
from datetime import datetime

import boto3
import requests
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

st.set_page_config(page_title="Cargar Imágenes", layout="wide")

st.title("Cargar Imágenes a S3")
st.markdown("Carga imágenes al bucket S3 cow-detect-maia y ejecutar detecció    n sobre las mismas")

S3_BUCKET = "cow-detect-maia"
# ENDPOINT_URL = "https://example.com"  # Configure your endpoint URL here
ENDPOINT_URL = "http://localhost:8000/predict"


@st.cache_resource
def get_s3_client():
    try:
        session = boto3.Session(profile_name="dvc-user")
        return session.client("s3")
    except NoCredentialsError:
        st.error("Credenciales AWS no encontradas. Por favor configure sus credenciales AWS.")
        return None


def upload_to_s3(file, s3_client, key):
    try:
        s3_client.upload_fileobj(file, S3_BUCKET, key)
        return True, f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            return False, f"El bucket {S3_BUCKET} no existe."
        elif error_code == "AccessDenied":
            return False, "Acceso denegado. Verifique sus permisos de AWS."
        else:
            return False, f"Error al cargar archivo: {str(e)}"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"


def call_endpoint(image_name, s3_path):
    assert ENDPOINT_URL is not None

    try:
        payload = {"name": image_name, "s3_path": f"s3://{S3_BUCKET}/{s3_path}"}
        response = requests.post(ENDPOINT_URL, json=payload, timeout=10)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, f"Error del endpoint: {str(e)}"


def main():
    s3_client = get_s3_client()

    if not s3_client:
        st.warning("Por favor configure las credenciales AWS para habilitar las cargas.")
        st.markdown("""
        ### Configuración AWS
        Puede configurar las credenciales AWS mediante:
        1. Variables de entorno: `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`
        2. Usando AWS CLI: `aws configure`
        3. Usando roles IAM si está ejecutando en infraestructura AWS
        """)
        return

    st.subheader("Configuración de Carga")

    col1, col2 = st.columns(2)

    with col1:
        finca = st.text_input(
            "Finca",
            value="",
            placeholder="ej. Andresalia",
            help="Nombre de la finca para organizar las cargas",
        )

    with col2:
        sobrevuelo = st.text_input(
            "Sobrevuelo",
            value="",
            placeholder="ej. 2025-09-01",
            help="Nombre del sobrevuelo/misión",
        )

    # Validar entradas
    if not finca or not sobrevuelo:
        st.warning("Por favor ingrese tanto Finca como Sobrevuelo para continuar")
        return

    # Construir la ruta del prefijo S3
    prefix = f"{finca}/{sobrevuelo}/" if finca and sobrevuelo else ""

    uploaded_files = st.file_uploader(
        "Seleccione imágenes para cargar",
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} archivo(s) seleccionado(s)")

        if st.button(
            "Cargar a S3 y Ejecutar Deteccion",
            type="primary",
            disabled=(not finca or not sobrevuelo),
        ):
            progress_bar = st.progress(0)
            status_container = st.container()

            successful_uploads = []
            failed_uploads = []
            endpoint_results = []
            endpoint_success_cnt = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                # Usar nombre de archivo original sin marca de tiempo para organización más limpia
                s3_key = f"{prefix}{uploaded_file.name}"

                uploaded_file.seek(0)
                success, message = upload_to_s3(uploaded_file, s3_client, s3_key)

                if success:
                    successful_uploads.append((uploaded_file.name, s3_key, message))

                    # Llamar endpoint
                    endpoint_result = call_endpoint(uploaded_file.name, s3_key)
                    if endpoint_result:
                        endpoint_success, endpoint_response = endpoint_result
                        endpoint_results.append(
                            (uploaded_file.name, endpoint_success, endpoint_response)
                        )
                        endpoint_success_cnt += endpoint_success
                else:
                    failed_uploads.append((uploaded_file.name, message))

                progress_bar.progress((idx + 1) / len(uploaded_files))

            with status_container:
                if successful_uploads:
                    st.success(f"Se cargaron exitosamente {len(successful_uploads)} archivo(s)")
                    with st.expander("Ver archivos cargados"):
                        for name, key, _url in successful_uploads:
                            st.text(f"{name} → s3://{S3_BUCKET}/{key}")

                if failed_uploads:
                    st.error(f"Fallo al cargar {len(failed_uploads)} archivo(s)")
                    with st.expander("Ver cargas fallidas"):
                        for name, error in failed_uploads:
                            st.text(f"{name}: {error}")

                if endpoint_success_cnt == 0:
                    st.error(
                        f"Fallo persistente llamando al end point de detección: {ENDPOINT_URL}"
                    )
                elif endpoint_success_cnt < len(successful_uploads):
                    success_ratio = endpoint_success_cnt / len(successful_uploads)
                    st.warning(f"Detección no fue completamente exitosa: {success_ratio:.2%}")
                else:
                    st.success("Detección exitosa para todas las imágenes")

                if endpoint_results:
                    with st.expander("Ver resultados del procesamiento"):
                        for name, success, response in endpoint_results:
                            if success:
                                st.success(f"{name}: Procesado exitosamente")
                                st.json(response)
                            else:
                                st.error(f"{name}: {response}")


if __name__ == "__main__":
    main()
