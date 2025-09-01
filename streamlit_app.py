import streamlit as st

st.set_page_config(
    page_title="Cow Detection App",
    layout="wide"
)

st.title("Cow Detection Application")
st.markdown("""
Welcome to the Cow Detection Application. Choose an option from the sidebar:

- **Upload Images**: Upload images to the S3 bucket for processing
- **View Detections**: Display images with detected bounding boxes
""")

st.sidebar.success("Select a page above.")