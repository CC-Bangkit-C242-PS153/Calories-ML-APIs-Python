from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from firestoredb import store_data
from datetime import datetime
import tensorflow as tf
import numpy as np
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

# Path model lokal
LOCAL_MODEL_PATH = "model/model.h5"

# Variabel global untuk menyimpan model
model = None
model_loaded = False


# Endpoint untuk meload model
@app.post("/load-model")
async def load_model(background_tasks: BackgroundTasks):
    global model, model_loaded

    if model_loaded:
        return {"message": "Model already loaded"}

    try:
        def load():
            global model, model_loaded
            print("Loading model...")
            model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
            model_loaded = True
            print("Model loaded successfully.")

        # Muat model di background untuk menghindari blocking
        background_tasks.add_task(load)
        return {"message": "Model loading started in the background"}
    except Exception as e:
        return {"message": f"Failed to start loading model: {str(e)}"}


# Endpoint prediksi
@app.post("/")
async def predict(request: Request):
    global model, model_loaded

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Call /load-model first.")

    try:
        payload = await request.json()
        # pubsubMessage = decode_base64_json(payload['message']['data'])
        pubsubMessage = payload

        new_data = np.array([[
            float(pubsubMessage['data']['water']),
            float(pubsubMessage['data']['protein']),
            float(pubsubMessage['data']['lipid']),
            float(pubsubMessage['data']['ash']),
            float(pubsubMessage['data']['carbohydrate']),
            float(pubsubMessage['data']['fiber']),
            float(pubsubMessage['data']['sugar']),
        ]])

        createdAt = datetime.now().isoformat()

        # Prediksi menggunakan model
        prediction = model.predict(new_data)
        print("Predicted Probabilities:", prediction[0])
        
        result = round(float(prediction[0][0]), 2)
        data = {
            "userId": pubsubMessage["userId"],
            "inferenceId": pubsubMessage["inferenceId"],
            "result": result,
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


def decode_base64_json(data):
    decoded_bytes = base64.b64decode(data)
    decoded_str = decoded_bytes.decode("utf-8")
    return json.loads(decoded_str)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
