import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import requests
from datetime import datetime
import os

st.set_page_config(page_title="Upload Images", layout="wide")

st.title("Upload Images to S3")
st.markdown("Upload images to the cow-detect-maia S3 bucket for processing.")

S3_BUCKET = "cow-detect-maia"
ENDPOINT_URL = "https://example.com"  # Configure your endpoint URL here

@st.cache_resource
def get_s3_client():
    try:
        return boto3.client('s3')
    except NoCredentialsError:
        st.error("AWS credentials not found. Please configure your AWS credentials.")
        return None

def upload_to_s3(file, s3_client, key):
    try:
        s3_client.upload_fileobj(file, S3_BUCKET, key)
        return True, f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            return False, f"Bucket {S3_BUCKET} does not exist."
        elif error_code == 'AccessDenied':
            return False, "Access denied. Check your AWS permissions."
        else:
            return False, f"Error uploading file: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def call_endpoint(image_name, s3_path):
    if not ENDPOINT_URL:
        return None

    try:
        payload = {
            "name": image_name,
            "s3_path": f"s3://{S3_BUCKET}/{s3_path}"
        }
        response = requests.post(ENDPOINT_URL, json=payload, timeout=10)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, f"Endpoint error: {str(e)}"

def main():
    s3_client = get_s3_client()

    if not s3_client:
        st.warning("Please configure AWS credentials to enable uploads.")
        st.markdown("""
        ### AWS Configuration
        You can configure AWS credentials by:
        1. Setting environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
        2. Using AWS CLI: `aws configure`
        3. Using IAM roles if running on AWS infrastructure
        """)
        return

    st.subheader("Upload Settings")

    col1, col2 = st.columns(2)

    with col1:
        farm = st.text_input(
            "Farm",
            value="",
            placeholder="e.g. farm-001",
            help="Name of the farm for organizing uploads"
        )

    with col2:
        flyover = st.text_input(
            "Flyover",
            value="",
            placeholder="e.g. survey-2024-01",
            help="Name of the flyover/mission"
        )

    # Validate inputs
    if not farm or not flyover:
        st.warning("Please enter both Farm and Flyover to proceed")

    # Build the S3 prefix path
    prefix = f"{farm}/{flyover}/" if farm and flyover else ""


    uploaded_files = st.file_uploader(
            "Choose images to upload",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True
        )

    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s)")

        if st.button("Upload to S3", type="primary", disabled=(not farm or not flyover)):
            progress_bar = st.progress(0)
            status_container = st.container()

            successful_uploads = []
            failed_uploads = []
            endpoint_results = []

            for idx, uploaded_file in enumerate(uploaded_files):
                # Use original filename without timestamp for cleaner organization
                s3_key = f"{prefix}{uploaded_file.name}"

                uploaded_file.seek(0)
                success, message = upload_to_s3(uploaded_file, s3_client, s3_key)

                if success:
                    successful_uploads.append((uploaded_file.name, s3_key, message))

                    # Call endpoint
                    endpoint_result = call_endpoint(uploaded_file.name, s3_key)
                    if endpoint_result:
                        endpoint_success, endpoint_response = endpoint_result
                        endpoint_results.append((uploaded_file.name, endpoint_success, endpoint_response))
                else:
                    failed_uploads.append((uploaded_file.name, message))

                progress_bar.progress((idx + 1) / len(uploaded_files))

            with status_container:
                if successful_uploads:
                    st.success(f"Successfully uploaded {len(successful_uploads)} file(s)")
                    with st.expander("View uploaded files"):
                        for name, key, url in successful_uploads:
                            st.text(f"{name} → s3://{S3_BUCKET}/{key}")

                if failed_uploads:
                    st.error(f"Failed to upload {len(failed_uploads)} file(s)")
                    with st.expander("View failed uploads"):
                        for name, error in failed_uploads:
                            st.text(f"{name}: {error}")

                if endpoint_results:
                    with st.expander("View processing results"):
                        for name, success, response in endpoint_results:
                            if success:
                                st.success(f"{name}: Processed successfully")
                                st.json(response)
                            else:
                                st.error(f"{name}: {response}")

if __name__ == "__main__":
    main()