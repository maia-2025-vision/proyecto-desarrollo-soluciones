import os
from datetime import datetime

import boto3
import requests
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

st.set_page_config(page_title="Cargar Imágenes", layout="wide")

st.title("Cargar Imágenes 📤")
st.markdown(
    "Carga imágenes y ejecutar detección sobre las mismas"
)  # Eliminé 'al bucket S3 cow-detect-maia' para simplificar la UI para el usuario final.

S3_BUCKET = "cow-detect-maia"
API_BASE_URL = os.getenv("APISERVICE_BASE_URL", "http://localhost:8000")
# ENDPOINT_URL = "https://example.com"  # Configura aquí la URL de tu endpoint
# PREDICT_ENDPOINT_URL = f"{API_BASE_URL}/predict" # Endpoint antiguo
PREDICTMANY_ENDPOINT_URL = (
    f"{API_BASE_URL}/predict-many"  # MODIFICADO: Usando el nuevo endpoint para lotes
)


@st.cache_resource
def get_s3_client():
    try:
        # Se elimina el perfil "hardcodeado". Boto3 buscará automáticamente
        # la variable de entorno AWS_PROFILE o usará el perfil 'default'.
        session = boto3.Session()
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


# COMENTADO: Antigua función para predecir una imagen a la vez.
# def call_endpoint(image_name, s3_path):
#     assert ENDPOINT_URL is not None
#
#     try:
#         payload = {"name": image_name, "s3_path": f"s3://{S3_BUCKET}/{s3_path}"}
#         response = requests.post(ENDPOINT_URL, json=payload, timeout=10)
#         response.raise_for_status()
#         return True, response.json()
#     except requests.exceptions.RequestException as e:
#         return False, f"Error del endpoint: {str(e)}"


#  Función para llamar al endpoint de lotes.
def call_batch_endpoint(s3_uris: list[str]):
    """Llama al endpoint de predicción por lotes con una lista de URIs de S3."""
    assert PREDICTMANY_ENDPOINT_URL is not None
    if not s3_uris:
        return False, "No se proporcionaron URIs de S3 al endpoint."

    try:
        payload = {"urls": s3_uris}
        # Timeout más corto porque se llama por lotes más pequeños
        response = requests.post(PREDICTMANY_ENDPOINT_URL, json=payload, timeout=60)
        response.raise_for_status()
        return True, response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        return False, f"Error del endpoint: {str(e)}"


def batches_from_list(uris: list[str], batch_size: int):
    """Genera lotes de un tamaño determinado a partir de una lista."""
    for start_idx in range(0, len(uris), batch_size):
        end_idx = start_idx + batch_size
        # Produce un batch de a lo más batch_size. Si el índice final > len(uris),
        # no hay problema, el batch simplemente será más pequeño.
        batch = uris[start_idx:end_idx]
        yield batch


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

    max_files = 20  # Para alcanzar a cubrir 5 bathces
    batch_size = 4

    uploaded_files = st.file_uploader(
        "Seleccione imágenes para cargar",
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        num_files = len(uploaded_files)
        st.info(f"{num_files} archivo(s) seleccionado(s)")

        if num_files > max_files:
            st.warning(
                f"Ha seleccionado {num_files} archivos. Por favor, seleccione "
                f"un máximo de {max_files} archivos a la vez."
            )
            upload_disabled = True
        else:
            upload_disabled = not finca or not sobrevuelo

        # Eliminé ' a S3' para simplificar el producto para el usuario final.
        if st.button(
            "Cargar y Ejecutar Detección",
            type="primary",
            disabled=upload_disabled,
        ):
            # --- Guardar Finca y Sobrevuelo en la sesión ---
            st.session_state["finca"] = finca
            st.session_state["sobrevuelo"] = sobrevuelo

            progress_bar = st.progress(0, text="Iniciando subida a S3...")
            status_container = st.container()

            successful_uploads = []
            failed_uploads = []
            s3_uris_for_api = []
            endpoint_results = []
            endpoint_success_cnt = 0

            #  Bucle de subida a S3
            for idx, uploaded_file in enumerate(uploaded_files):
                # Eliminé 'a S3' para simplificar el producto para el usuario final.
                progress_text_s3 = f"Subiendo: {uploaded_file.name} ({idx + 1}/{num_files})"
                progress_bar.progress((idx + 1) / num_files * 0.5, text=progress_text_s3)

                s3_key = f"{prefix}{uploaded_file.name}"
                uploaded_file.seek(0)
                success, message = upload_to_s3(uploaded_file, s3_client, s3_key)

                if success:
                    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
                    successful_uploads.append((uploaded_file.name, s3_key, message))
                    s3_uris_for_api.append(s3_uri)
                else:
                    failed_uploads.append((uploaded_file.name, message))

            #  Bucle de procesamiento en lotes con la API
            if s3_uris_for_api:
                # Crear un mapa de URI -> nombre de archivo para una búsqueda eficiente O(n)
                # Esto evita tener que buscar en la lista 'successful_uploads' en cada iteración
                uri_to_name_map = {f"s3://{S3_BUCKET}/{k}": n for n, k, _ in successful_uploads}

                # --- Lógica de lotes exactamente como la sugirió cuckookernel ---
                all_batch_responses = []
                processed_images_count = 0

                for batch in batches_from_list(s3_uris_for_api, batch_size=batch_size):
                    num_in_batch = len(batch)

                    with st.spinner(f"Procesando un lote de {num_in_batch} imágenes."):
                        # Le quite 'imágenes con el API...'Porque es el producto para el usuario
                        # final que no necesita saber esto.
                        # Si les parece podemos agregarlo nuevamente.
                        batch_success, batch_response_increment = call_batch_endpoint(batch)

                    if batch_success:
                        all_batch_responses.extend(batch_response_increment)
                    else:
                        st.error(
                            f"La llamada al API falló para un lote: {batch_response_increment}"
                        )
                        # Marcar todas las imágenes de este lote como fallidas
                        for uri in batch:
                            name = uri_to_name_map.get(uri, "NombreDesconocido")
                            endpoint_results.append(
                                (name, False, f"Fallo en el lote: {batch_response_increment}")
                            )

                    # Actualizar progreso basado en el número de imágenes procesadas
                    processed_images_count += num_in_batch
                    progress_percentage = 0.5 + (
                        processed_images_count / len(s3_uris_for_api) * 0.5
                    )
                    progress_text_api = (
                        f"Procesadas {processed_images_count}/{len(s3_uris_for_api)} imágenes..."
                    )
                    progress_bar.progress(progress_percentage, text=progress_text_api)

                #  Procesamiento final de todos los resultados acumulados
                if all_batch_responses:
                    response_map = {res.get("url"): res for res in all_batch_responses}
                    for uri in s3_uris_for_api:
                        name = uri_to_name_map.get(uri)
                        if name and uri in response_map:
                            endpoint_results.append((name, True, response_map[uri]))
                            endpoint_success_cnt += 1
                        elif name and not any(uri in r.get("url", "") for r in all_batch_responses):
                            # Esto captura imágenes que estaban en lotes fallidos
                            continue
                        elif name:
                            endpoint_results.append(
                                (name, False, "No se recibió respuesta del API para esta imagen.")
                            )

            progress_bar.progress(1.0, text="¡Proceso completado!")

            with status_container:
                if successful_uploads:
                    st.success(f"Se cargaron exitosamente {len(successful_uploads)} archivo(s)")
                    with st.expander("Ver archivos cargados"):
                        for name, key, _ in successful_uploads:
                            st.text(f"{name} → s3://{S3_BUCKET}/{key}")

                if failed_uploads:
                    st.error(f"Fallo al cargar {len(failed_uploads)} archivo(s)")
                    with st.expander("Ver cargas fallidas"):
                        for name, error in failed_uploads:
                            st.text(f"{name}: {error}")

                if endpoint_success_cnt == 0:
                    st.error(
                        f"Fallo persistente llamando al end point de "
                        f"detección: {PREDICTMANY_ENDPOINT_URL}"
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

            # --- Almacenar resultados en el historial de la sesión ---
            if "detection_history" not in st.session_state:
                st.session_state["detection_history"] = []

            if endpoint_results:
                for name, success, response in endpoint_results:
                    if success:
                        s3_key = next((k for n, k, u in successful_uploads if n == name), None)
                        if s3_key:
                            # Añadir el resultado al historial persistente
                            st.session_state["detection_history"].append(
                                {
                                    "name": name,
                                    "s3_uri": f"s3://{S3_BUCKET}/{s3_key}",
                                    "detections": response.get("detections", {}),
                                    "finca": finca,
                                    "sobrevuelo": sobrevuelo,
                                }
                            )

            st.toast(f"¡Procesamiento completado para {len(endpoint_results)} imágenes!", icon="🎉")


if __name__ == "__main__":
    main()
