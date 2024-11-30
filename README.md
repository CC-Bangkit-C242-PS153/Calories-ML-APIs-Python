# Fitcal Calories ML Model Backend

This repository contains the backend implementation and act as MLs Endpoint that's call at core APIs through pub/sub as message broker for the Fitcal app.

## Project Description

The Fitcal backend is built using the Python programming language, and FastAPI framework. It provides API endpoints for user-related operations. The backend utilizes Firebase Auth Token for authentication, and Firestore for database NoSQL.

## Getting Started

To get started with the Fitcal MLs backend, follow these steps:

1. Clone the repository: `git clone https://github.com/CC-Bangkit-C242-PS153/Calories-ML-APIs-Python.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Set up the environment variables by creating a `.env` file (refer to the .env section below).
4. Run the application: `python main.py`


## Dependencies

```
fastapi==0.115.5
google-cloud-pubsub==2.27.1
google-cloud-storage==2.18.2
pillow==11.0.0
tensorflow==2.18.0
uvicorn==0.32.0
```

## Environment Variables

The following environment variables are required to run the Fitcal backend:

- `BUCKET_NAME`: cloud storage bucket name.

Make sure to set these variables in the `.env` file before running the application.

## Project Structure
```bash
├── README.md
├── .gitignore
├── main.py
├── storage.py
├── requirements.txt
├── model
│   └── model.h5
```
