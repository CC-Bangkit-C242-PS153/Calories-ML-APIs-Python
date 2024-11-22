from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from firestoredb import store_data
from storage import download
from datetime import datetime
import tensorflow as tf
import numpy as np
import asyncio
import os
import base64
import json

# Membuat aplikasi FastAPI
app = FastAPI()

# Menambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan domain yang dibutuhkan
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode HTTP
    allow_headers=["*"],  # Izinkan semua header
)

def decode_base64_json(data):
    # Mendekode data Base64 menjadi bytes
    decoded_bytes = base64.b64decode(data)
    
    # Mengonversi bytes menjadi string (UTF-8) dan kemudian parsing JSON
    decoded_str = decoded_bytes.decode('utf-8')
    return json.loads(decoded_str)

# Variabel global untuk menyimpan model
model = None
LOCAL_MODEL_PATH = "model/model.h5"

# Fungsi untuk memuat model secara asinkron
async def load_model_async():
    global model
    try:
        print("Loading model asynchronously from local storage...")
        # Operasi sinkron dimasukkan ke thread pool agar kompatibel dengan asyncio
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, tf.keras.models.load_model, LOCAL_MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

# Dependency untuk inisialisasi model
async def get_model():
    global model
    if model is None:
        await load_model_async()  # Pastikan model dimuat jika belum ada
    return model

@app.post("/")
async def home(request: Request, model: tf.keras.Model = Depends(get_model)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak tersedia")
    
    try:
        payload = await request.json()
        pubsubMessage = decode_base64_json(payload['message']['data'])
        image = download(pubsubMessage)
        print(image)

        new_data = np.array([
            [
                float(pubsubMessage['data']['water']),
                float(pubsubMessage['data']['protein']),
                float(pubsubMessage['data']['lipid']),
                float(pubsubMessage['data']['ash']),
                float(pubsubMessage['data']['carbohydrate']),
                float(pubsubMessage['data']['fiber']),
                float(pubsubMessage['data']['sugar']),
            ]
        ])

        createdAt = datetime.now().isoformat()

        prediction = model.predict(new_data)
        print(f"Data:")
        print("Predicted Probabilities:", prediction[0])
        print(f"{prediction}")
        
        result = round(float(prediction[0][0]),2)
        data = {
            "userId": pubsubMessage["userId"],
            "inferenceId": pubsubMessage["inferenceId"],
            "result": result,
            "statusImage":image,
            "createdAt": createdAt,
        }

        store_data(pubsubMessage["userId"], pubsubMessage["inferenceId"], data)
        
        return JSONResponse(
            status_code=201,
            content={
                "status": "Success",
                "statusCode": 201,
                "message": "Successfully to do inference",
                "data": data,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "Fail to do Inference",
                "statusCode": 400,
                "message": f"Error: {e}",
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
