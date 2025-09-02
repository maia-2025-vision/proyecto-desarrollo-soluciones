import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import json
from typing import Dict, List, Any, Optional
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# --- AWS S3 Client (Minimal) ---
@st.cache_resource
def get_s3_client():
    """Initializes a boto3 S3 client, caching the resource."""
    try:
        # Assumes credentials are configured in the environment (e.g., ~/.aws/credentials)
        return boto3.client('s3')
    except NoCredentialsError:
        st.error("Credenciales de AWS no encontradas. No se podrán cargar imágenes desde S3.")
        return None
    except Exception as e:
        st.error(f"No se pudo crear el cliente S3: {e}")
        return None

# --- Image Loading Logic ---
def load_image(img_data: dict):
    """Loads an image from a local path or an S3 URI based on the provided data."""
    if "image_path" in img_data:  # Local preview case from 1_Upload_Images.py
        try:
            return Image.open(img_data["image_path"])
        except FileNotFoundError:
            st.error(f"Archivo de vista previa no encontrado: {img_data['image_path']}")
            return None
    
    elif "s3_uri" in img_data:  # S3/API case
        s3_client = get_s3_client()
        if not s3_client: return None
        try:
            bucket, key = img_data["s3_uri"].replace("s3://", "").split("/", 1)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return Image.open(io.BytesIO(response['Body'].read()))
        except ClientError as e:
            st.error(f"Error de acceso a S3 ({img_data['s3_uri']}): {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error inesperado cargando desde S3: {e}")
            return None
            
    st.warning(f"No se encontró 'image_path' o 's3_uri' para: {img_data.get('name')}")
    return None

# --- Drawing Logic ---
def draw_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draws bounding boxes and labels on a copy of the image."""
    img_with_boxes = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_with_boxes)
    try:
        # A commonly available font
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    for det in detections:
        box = det.get("bbox")
        if isinstance(box, list) and len(box) == 4:
            draw.rectangle(box, outline="red", width=3)
            score = det.get("score", 0.0)
            label = f"{det.get('label', 'vaca')}: {score:.2f}"
            draw.text((box[0], box[1] - 25), label, fill="red", font=font)
    
    return img_with_boxes

# --- Main Page UI ---
def main():
    st.set_page_config(page_title="Ver Detecciones", layout="wide")
    st.title("Visualizador de Detecciones")

    # The single source of truth is the session state populated by the upload page
    if "detection_data" in st.session_state and st.session_state.detection_data.get("images"):
        images_to_display = st.session_state.detection_data["images"]
        
        finca = st.session_state.get('finca', 'N/A')
        sobrevuelo = st.session_state.get('sobrevuelo', 'N/A')
        st.header(f"Resultados para Finca: {finca} | Sobrevuelo: {sobrevuelo}")
        st.info(f"Se encontraron {len(images_to_display)} imágenes para mostrar.")
        
        for img_data in images_to_display:
            st.markdown("---")
            st.subheader(f"Imagen: {img_data.get('name', 'Nombre no disponible')}")
            
            image = load_image(img_data)
            
            if image:
                detections = img_data.get("detections", [])
                if detections:
                    st.image(draw_bounding_boxes(image, detections), use_container_width=True)
                    with st.expander("Ver datos de detección (JSON)"):
                        st.json(detections)
                else:
                    st.image(image, use_container_width=True)
                    st.caption("No se encontraron detecciones para esta imagen.")
            else:
                st.error("No se pudo cargar o mostrar la imagen.")
                with st.expander("Ver datos del intento de carga"):
                    st.json(img_data)
    else:
        st.info("Aún no hay datos para mostrar. Por favor, vaya a la página de carga para procesar imágenes.")
        st.page_link("pages/1_Upload_Images.py", label="Ir a Cargar Imágenes", icon="⬆️")

if __name__ == "__main__":
    main()