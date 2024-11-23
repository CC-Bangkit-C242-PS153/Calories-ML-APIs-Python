from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from firestoredb import store_data
from storage import download
from datetime import datetime
import tensorflow as tf
import numpy as np
import threading
import os
import base64
import json
import asyncio

# Membuat aplikasi FastAPI
app = FastAPI()
LOCAL_MODEL_PATH = "model/model.h5"
model = None
model_loaded = False
load_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Fungsi untuk menunggu hingga model diload
async def wait_for_model_to_load(timeout: int = 30):
    global model_loaded
    elapsed_time = 0
    check_interval = 1  # Periksa setiap 1 detik
    
    while not model_loaded:
        if elapsed_time >= timeout:
            raise HTTPException(status_code=503, detail="Model not loaded within the expected time.")
        await asyncio.sleep(check_interval)
        elapsed_time += check_interval

def load():
            with load_lock:
                global model, model_loaded
                print("Loading model...")
                model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
                model_loaded = True
                print("Model loaded successfully.")

# Endpoint untuk meload model
@app.post("/load-model")
async def load_model(background_tasks: BackgroundTasks):
    global model, model_loaded

    if model_loaded:
        return JSONResponse(
            status_code=200,
            content={
                 "status": "Success",
                "statusCode": 200,
                "message": "Model already loaded",
            }
        )

    try:
        load()
        # Muat model di background untuk menghindari blocking
        background_tasks.add_task(load)
        return JSONResponse(
            status_code=202,  # Accepted, karena proses dilakukan di background
            content={
                "status": "Success",
                "statusCode": 200,
                "message": "Successfully Load Model",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,  # Internal server error
            content={
                 "status": "Failed",
                 "statusCode": 500,
                 "message": f"Failed to start loading model: {str(e)}"
                 }
        )


# Endpoint prediksi
@app.post("/")
async def predict(request: Request):
    global model, model_loaded

    while not model_loaded:
         load()

    # Tunggu hingga model diload
    await wait_for_model_to_load()

    try:
        payload = await request.json()
        pubsubMessage = decode_base64_json(payload['message']['data'])
        image = download(pubsubMessage)

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
            "statusImage":image,
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
