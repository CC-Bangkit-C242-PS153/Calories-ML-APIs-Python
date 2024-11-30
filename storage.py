from google.cloud import storage
import os

# Inisialisasi klien Google Cloud Storage
storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME")

def download(data):
    """Mengunduh file dari Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    file_path = f"prediction/{data['inferenceId']}.{data['data']['type']['ext']}"
    blob = bucket.blob(file_path)

    # Mengunduh konten file
    contents = blob.download_as_bytes()  # Konten dalam format bytes
    return contents
