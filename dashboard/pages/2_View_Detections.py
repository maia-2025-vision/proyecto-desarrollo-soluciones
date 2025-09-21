import io

import streamlit as st
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from api.s3_utils import download_file_from_s3


# --- Lógica de Carga de Imágenes (Refactorizada) ---
def load_image(img_data: dict) -> Image.Image | None:
    """Carga una imagen desde una S3 URI usando una función de utilidad centralizada."""
    s3_uri = img_data.get("s3_uri")
    if not s3_uri:
        st.warning(f"No se encontró 's3_uri' para: {img_data.get('name')}")
        return None

    try:
        image_bytes = download_file_from_s3(s3_uri)
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Error al cargar la imagen desde {s3_uri}: {e}")
        return None


# --- Drawing Logic ---
def draw_bounding_boxes(
    image: Image.Image, detections: list, confidence_threshold: float
) -> Image.Image:
    """Dibuja cuadros delimitadores y un fondo translúcido para el texto del puntaje."""
    img_with_boxes = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_with_boxes)

    font_size = 60
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        # Fallbacks de fuentes
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
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

                # score_text = f"{score:.2f}"  # Decimal (ej. 0.95)
                score_text = f"{score * 100:.0f}%"  # Porcentaje (ej. 95%)
                text_position = (box[0], box[1] - (font_size + 5))

                # --- Lógica para fondo translúcido nítido ---
                try:
                    text_bbox = draw.textbbox(text_position, score_text, font=font)
                    padding = 10
                    bg_bbox = (
                        int(text_bbox[0] - padding),
                        int(text_bbox[1] - padding),
                        int(text_bbox[2] + padding),
                        int(text_bbox[3] + padding),
                    )

                    # Crear un panel blanco translúcido
                    overlay = Image.new("RGBA", img_with_boxes.size, (255, 255, 255, 0))
                    draw_overlay = ImageDraw.Draw(overlay)
                    draw_overlay.rectangle(
                        bg_bbox, fill=(255, 255, 255, 60)
                    )  # Blanco con ~76% de transparencia

                    # Combinar la capa de superposición con la imagen principal
                    img_with_boxes = Image.alpha_composite(img_with_boxes, overlay)
                    draw = ImageDraw.Draw(img_with_boxes)  # Volver a crear el objeto Draw

                except Exception as e:
                    # En lugar de un 'pass' silencioso, registramos una advertencia.
                    logger.warning(
                        f"No se pudo dibujar el efecto de fondo traslúcido para el score: {e}"
                    )

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

    confidence_percent = st.sidebar.slider(
        "Umbral de Confianza", min_value=0, max_value=100, value=50, step=5, format="%d%%"
    )
    confidence_threshold = confidence_percent / 100.0

    st.title("Visualizador de Detecciones🖼️")

    # Leer el historial de detecciones persistente de la sesión
    if "detection_history" in st.session_state and st.session_state.detection_history:
        history = st.session_state.detection_history

        # --- FILTROS POR FINCA Y SOBREVUELO ---
        st.sidebar.divider()
        st.sidebar.subheader("Filtrar Resultados")

        fincas = sorted({item["finca"] for item in history})
        selected_finca = st.sidebar.selectbox("Seleccionar Finca", ["Todas"] + fincas)

        if selected_finca and selected_finca != "Todas":
            sobrevuelos = sorted(
                {item["sobrevuelo"] for item in history if item["finca"] == selected_finca}
            )
            selected_sobrevuelo = st.sidebar.selectbox(
                "Seleccionar Sobrevuelo", ["Todos"] + sobrevuelos
            )
        else:
            selected_sobrevuelo = "Todos"

        # --- Filtrar imágenes para mostrar ---
        images_to_display = history
        if selected_finca != "Todas":
            images_to_display = [
                item for item in images_to_display if item["finca"] == selected_finca
            ]
        if selected_sobrevuelo != "Todos":
            images_to_display = [
                item for item in images_to_display if item["sobrevuelo"] == selected_sobrevuelo
            ]

        if not images_to_display:
            st.warning("No se encontraron imágenes con los filtros seleccionados.")
        else:
            st.info(f"Mostrando {len(images_to_display)} imagen(es) con los filtros seleccionados.")
            st.divider()

            # Bucle para mostrar las imágenes filtradas
            for img_data in images_to_display:
                st.divider()
                finca = img_data.get("finca", "N/A")
                sobrevuelo = img_data.get("sobrevuelo", "N/A")
                st.subheader(
                    f"Imagen: {img_data.get('name', 'N/A')} "
                    f"(Finca: {finca} | Sobrevuelo: {sobrevuelo})"
                )

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

                        annotated_image = draw_bounding_boxes(
                            image, reformatted_detections, confidence_threshold
                        )

                        st.image(annotated_image, width="stretch")

                        with st.expander("Ver datos de detección (JSON)"):
                            st.json(detections_dict)

                        st.download_button(
                            label="Descargar imagen anotada",
                            data=image_to_bytes(annotated_image),
                            file_name=f"annotated_{img_data.get('name', 'image.png')}",
                            mime="image/png",
                            key=f"download_{img_data.get('name')}",
                        )
                    else:
                        st.image(image, width="stretch")
                        st.caption("No se encontraron detecciones para esta imagen.")
                else:
                    st.error("No se pudo cargar o mostrar la imagen.")
                    with st.expander("Ver datos del intento de carga"):
                        st.json(img_data)

    else:
        st.info(
            "Aún no hay datos para mostrar. Por favor, vaya a la página "
            "de carga para procesar imágenes."
        )
        st.page_link("pages/1_Upload_Images.py", label="Ir a Cargar Imágenes", icon="⬆️")


if __name__ == "__main__":
    main()
