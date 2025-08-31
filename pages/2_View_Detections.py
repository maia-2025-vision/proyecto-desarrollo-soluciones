import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import json
from typing import Dict, List, Any, Optional
import numpy as np

st.set_page_config(page_title="View Detections", layout="wide")

st.title("View Detection Results")
st.markdown("Display images with bounding boxes from detection API endpoint.")

def fetch_detection_data(endpoint_url: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(endpoint_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON response: {str(e)}")
        return None

def load_image_from_url(image_url: str) -> Optional[Image.Image]:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading image from {image_url}: {str(e)}")
        return None

def load_image_from_path(image_path: str) -> Optional[Image.Image]:
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image from {image_path}: {str(e)}")
        return None

def draw_bounding_boxes(
    image: Image.Image, 
    boxes: List[Dict[str, Any]], 
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    box_color: str = "red",
    text_color: str = "white",
    line_width: int = 3
) -> Image.Image:
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for idx, box in enumerate(boxes):
        if isinstance(box, dict):
            if 'xmin' in box and 'ymin' in box and 'xmax' in box and 'ymax' in box:
                x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            elif 'x' in box and 'y' in box and 'width' in box and 'height' in box:
                x1, y1 = box['x'], box['y']
                x2, y2 = x1 + box['width'], y1 + box['height']
            else:
                continue
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = box
        else:
            continue
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=line_width)
        
        label_text = ""
        if labels and idx < len(labels):
            label_text = str(labels[idx])
        if scores and idx < len(scores):
            score_text = f"{scores[idx]:.2f}" if isinstance(scores[idx], float) else str(scores[idx])
            label_text = f"{label_text} ({score_text})" if label_text else score_text
        
        if label_text:
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.rectangle(
                [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)],
                fill=box_color
            )
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=text_color, font=font)
    
    return img_with_boxes

def main():
    st.sidebar.header("Configuration")
    
    input_method = st.sidebar.radio(
        "Input Method",
        ["API Endpoint", "Sample JSON"]
    )
    
    if input_method == "API Endpoint":
        endpoint_url = st.sidebar.text_input(
            "API Endpoint URL",
            placeholder="https://api.example.com/detections",
            help="Enter the URL of your detection API endpoint"
        )
        
        if st.sidebar.button("Fetch Detections"):
            if endpoint_url:
                with st.spinner("Fetching detection data..."):
                    data = fetch_detection_data(endpoint_url)
                    if data:
                        st.session_state['detection_data'] = data
                        st.success("Data fetched successfully!")
            else:
                st.warning("Please enter an API endpoint URL")
    
    else:
        st.sidebar.markdown("### Sample JSON Format")
        sample_json = st.sidebar.text_area(
            "Paste JSON data",
            value=json.dumps({
                "images": [
                    {
                        "image_url": "https://example.com/image1.jpg",
                        "detections": [
                            {
                                "xmin": 100, "ymin": 100, 
                                "xmax": 200, "ymax": 200,
                                "label": "cow",
                                "confidence": 0.95
                            }
                        ]
                    }
                ]
            }, indent=2),
            height=300
        )
        
        if st.sidebar.button("Load JSON"):
            try:
                data = json.loads(sample_json)
                st.session_state['detection_data'] = data
                st.success("JSON loaded successfully!")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    
    box_color = st.sidebar.color_picker("Box Color", "#FF0000")
    text_color = st.sidebar.color_picker("Text Color", "#FFFFFF")
    line_width = st.sidebar.slider("Line Width", 1, 10, 3)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
    
    if 'detection_data' in st.session_state:
        data = st.session_state['detection_data']
        
        images_data = []
        if 'images' in data:
            images_data = data['images']
        elif 'results' in data:
            images_data = data['results']
        elif isinstance(data, list):
            images_data = data
        else:
            images_data = [data]
        
        if not images_data:
            st.warning("No images found in the data")
            return
        
        st.subheader(f"Detection Results ({len(images_data)} images)")
        
        cols_per_row = st.slider("Images per row", 1, 4, 2)
        
        for i in range(0, len(images_data), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(images_data):
                    img_data = images_data[i + j]
                    
                    with col:
                        image = None
                        if 'image_url' in img_data:
                            image = load_image_from_url(img_data['image_url'])
                        elif 'image_path' in img_data:
                            image = load_image_from_path(img_data['image_path'])
                        elif 'image' in img_data:
                            if isinstance(img_data['image'], str):
                                if img_data['image'].startswith('http'):
                                    image = load_image_from_url(img_data['image'])
                                else:
                                    image = load_image_from_path(img_data['image'])
                        
                        if image:
                            detections = []
                            labels = []
                            scores = []
                            
                            if 'detections' in img_data:
                                detections = img_data['detections']
                            elif 'boxes' in img_data:
                                detections = img_data['boxes']
                            elif 'bounding_boxes' in img_data:
                                detections = img_data['bounding_boxes']
                            
                            for det in detections:
                                if 'label' in det:
                                    labels.append(det['label'])
                                elif 'class' in det:
                                    labels.append(det['class'])
                                elif 'category' in det:
                                    labels.append(det['category'])
                                else:
                                    labels.append("object")
                                
                                if show_confidence:
                                    if 'confidence' in det:
                                        scores.append(det['confidence'])
                                    elif 'score' in det:
                                        scores.append(det['score'])
                                    elif 'prob' in det:
                                        scores.append(det['prob'])
                            
                            if detections:
                                image_with_boxes = draw_bounding_boxes(
                                    image, 
                                    detections, 
                                    labels=labels if labels else None,
                                    scores=scores if scores and show_confidence else None,
                                    box_color=box_color,
                                    text_color=text_color,
                                    line_width=line_width
                                )
                                st.image(image_with_boxes, use_container_width=True)
                                st.caption(f"Detections: {len(detections)}")
                            else:
                                st.image(image, use_container_width=True)
                                st.caption("No detections")
                            
                            with st.expander("Detection Details"):
                                for idx, det in enumerate(detections):
                                    st.json(det)
                        else:
                            st.error(f"Could not load image {i + j + 1}")
    else:
        st.info("👈 Use the sidebar to load detection data from an API endpoint or paste sample JSON")
        
        st.markdown("""
        ### Expected JSON Format
        
        The API should return JSON in one of these formats:
        
        **Format 1: Images with detections**
        ```json
        {
            "images": [
                {
                    "image_url": "https://example.com/image.jpg",
                    "detections": [
                        {
                            "xmin": 100, "ymin": 100,
                            "xmax": 200, "ymax": 200,
                            "label": "cow",
                            "confidence": 0.95
                        }
                    ]
                }
            ]
        }
        ```
        
        **Format 2: Alternative box format**
        ```json
        {
            "results": [
                {
                    "image_path": "/path/to/image.jpg",
                    "boxes": [
                        {
                            "x": 100, "y": 100,
                            "width": 100, "height": 100,
                            "class": "cow",
                            "score": 0.95
                        }
                    ]
                }
            ]
        }
        ```
        """)

if __name__ == "__main__":
    main()