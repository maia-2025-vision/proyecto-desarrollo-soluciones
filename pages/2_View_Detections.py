import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import json
from typing import Dict, List, Any, Optional
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import zipfile 

# --- AWS S3 Client (Minimal) ---
@st.cache_resource
def get_s3_client():
    """Inicializa un cliente de S3 de boto3."""
    try:
        #Se especifica el perfil para asegurar que se usen las credenciales correctas.
        session = boto3.Session(profile_name="dvc-user")
        return session.client("s3")
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
def draw_bounding_boxes(image: Image.Image, detections: list, confidence_threshold: float) -> Image.Image:
    """Dibuja cuadros delimitadores y un fondo translúcido para el texto del puntaje."""
    img_with_boxes = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_with_boxes)

    font_size = 60
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        # Fallbacks de fuentes
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.load_default(size=font_size)
            except AttributeError:
                font = ImageFont.load_default()

    for det in detections:
        score = det.get("score", 0.0)
        if score >= confidence_threshold:
            box = det.get("bbox")
            
            if score >= 0.8:
                color = "green"
            elif score >= 0.5:
                color = "yellow"
            else:
                color = "red"

            if isinstance(box, list) and len(box) == 4:
                draw.rectangle(box, outline=color, width=5)
                
                score_text = f"{score:.2f}"
                text_position = (box[0], box[1] - (font_size + 5))
                
                # --- Lógica para fondo translúcido nítido ---
                try:
                    text_bbox = draw.textbbox(text_position, score_text, font=font)
                    padding = 10
                    bg_bbox = (
                        int(text_bbox[0] - padding), int(text_bbox[1] - padding),
                        int(text_bbox[2] + padding), int(text_bbox[3] + padding)
                    )
                    
                    # Crear un panel blanco translúcido
                    overlay = Image.new('RGBA', img_with_boxes.size, (255, 255, 255, 0))
                    draw_overlay = ImageDraw.Draw(overlay)
                    draw_overlay.rectangle(bg_bbox, fill=(255, 255, 255, 60)) # Blanco con ~76% de transparencia

                    # Combinar la capa de superposición con la imagen principal
                    img_with_boxes = Image.alpha_composite(img_with_boxes, overlay)
                    draw = ImageDraw.Draw(img_with_boxes) # Volver a crear el objeto Draw

                except Exception:
                    # Fallback si falla el efecto
                    pass
                
                # Dibujar el texto encima del área procesada
                draw.text(text_position, score_text, font=font, fill=color)

    return img_with_boxes.convert("RGB")


def image_to_bytes(image: Image.Image) -> bytes:
    """Converts a PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# --- Main Page UI ---
def main():
    st.set_page_config(page_title="Ver Detecciones", layout="wide")
    st.sidebar.title("Opciones de Visualización")
    confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza", 0.0, 1.0, 0.5, 0.05
    )
    
    st.title("Visualizador de Detecciones🖼️")

    # Leer el historial de detecciones persistente de la sesión
    if "detection_history" in st.session_state and st.session_state.detection_history:
        images_to_display = st.session_state.detection_history
        
        st.info(f"Se encontraron {len(images_to_display)} imágenes en el historial para mostrar.")
        
        # --- Lógica para preparar el botón de Descargar Todo (ZIP) ---
        # (Se mantiene la lógica eficiente de un solo bucle)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for img_data in images_to_display:
                st.markdown("---")
                # Mostrar Finca y Sobrevuelo junto al nombre de la imagen
                finca = img_data.get('finca', 'N/A')
                sobrevuelo = img_data.get('sobrevuelo', 'N/A')
                st.subheader(f"Imagen: {img_data.get('name', 'N/A')} (Finca: {finca} | Sobrevuelo: {sobrevuelo})")
                
                image = load_image(img_data)
                
                if image:
                    detections_dict = img_data.get("detections", {})
                    if detections_dict and "boxes" in detections_dict:
                        # Reformatear los datos de detección
                        boxes = detections_dict.get("boxes", [])
                        scores = detections_dict.get("scores", [0.0] * len(boxes))
                        labels = detections_dict.get("labels", ["vaca"] * len(boxes))
                        
                        reformatted_detections = [
                            {"bbox": box, "score": scores[i], "label": labels[i]}
                            for i, box in enumerate(boxes)
                        ]

                        annotated_image = draw_bounding_boxes(image, reformatted_detections, confidence_threshold)
                        
                        # 1. Display the annotated image
                        st.image(annotated_image, width='stretch')

                        # 2. Add the annotated image to the zip file in memory
                        zip_file.writestr(
                            f"{finca}_{sobrevuelo}_{img_data.get('name', 'image.png')}", 
                            image_to_bytes(annotated_image)
                        )

                        with st.expander("Ver datos de detección (JSON)"):
                            st.json(detections_dict)
                        
                        st.download_button(
                           label="Descargar imagen anotada",
                           data=image_to_bytes(annotated_image),
                           file_name=f"annotated_{img_data.get('name', 'image.png')}",
                           mime="image/png",
                           key=f"download_{img_data.get('name')}" # Añadida key única
                        )
                    else:
                        st.image(image, width='stretch')
                        st.caption("No se encontraron detecciones para esta imagen.")
                else:
                    st.error("No se pudo cargar o mostrar la imagen.")
                    with st.expander("Ver datos del intento de carga"):
                        st.json(img_data)
        
        # This is now rendered after the loop has populated the zip buffer.
        st.sidebar.download_button(
            label="Descargar Todo (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="resultados_deteccion.zip",
            mime="application/zip",
        )

    else:
        st.info("Aún no hay datos para mostrar. Por favor, vaya a la página de carga para procesar imágenes.")
        st.page_link("pages/1_Upload_Images.py", label="Ir a Cargar Imágenes", icon="⬆️")

if __name__ == "__main__":
    main()