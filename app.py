import os
import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 1️⃣ MODEL PATHS & KEYS
# ======================================================
MODEL_PATH = "model.tflite"
CLASSES_PATH = "class_indices.json"
IMAGE_SIZE = (224, 224)
API_KEY = "OPENROUTER_API_KEY"
# <-- replace with your OpenRouter key

# Load class indices and TFLite model
try:
    leaf_detector = YOLO("best.pt")     # Your downloaded YOLO model
    print("✅ Leaf detector loaded.")
except Exception as e:
    print("❌ Error loading leaf detector:", e)
    leaf_detector = None
try:
    # Load class indices
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ Disease model loaded.")
except Exception as e:
    print("❌ Error loading disease model:", e)
    interpreter, idx_to_class = None, None


# ======================================================
# 3️⃣ IMAGE PREPROCESSING
# ======================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ======================================================
# 4️⃣ YOLO LEAF DETECTION
# ======================================================
def detect_leaf(image):
    """
    Returns True if YOLO detects a leaf, else False.
    """
    try:
        results = leaf_detector.predict(image, conf=0.50)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return False
        return True

    except Exception as e:
        print("YOLO detection error:", e)
        return False


# ======================================================
# 5️⃣ DISEASE PREDICTION USING TFLITE
# ======================================================
def predict_disease(interpreter, input_array, idx_to_class):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(preds))
    return idx_to_class.get(idx, "Unknown"), float(preds[idx]) * 100


# ======================================================
# 6️⃣ OPENROUTER RECOMMENDATION
# ======================================================
async def get_openrouter_recommendation(disease_name: str) -> str:
    prompt = (
        f"Give simple treatment steps for {disease_name} in farmer-friendly language. "
        "Use bullet points and put each step on a new line."
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    loop = asyncio.get_event_loop()

    def send_request():
        return requests.post(url, headers=headers, json=data)

    try:
        response = await loop.run_in_executor(None, send_request)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"Could not generate recommendation (Error {response.status_code})"

    except Exception as e:
        return f"Could not generate recommendation: {str(e)}"


# ======================================================
# 7️⃣ FINAL API ENDPOINT
# ======================================================
@app.post("/predict-disease")
async def predict_disease_api(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Disease model not loaded.")

    if leaf_detector is None:
        raise HTTPException(status_code=500, detail="Leaf detector model not loaded.")

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Step 1: Detect leaf using YOLO
        leaf_present = detect_leaf(image)

        if not leaf_present:
            return {
                "status": "error",
                "leaf_detected": False,
                "message": "No leaf detected. Upload a clear leaf image."
            }

        # Step 2: Disease prediction
        input_arr = preprocess_image(image)
        disease, conf = predict_disease(interpreter, input_arr, idx_to_class)

        # Step 3: Get recommendation
        recommendation = await get_openrouter_recommendation(disease)

        return {
            "status": "success",
            "leaf_detected": True,
            "predicted_disease": disease,
            "confidence": f"{conf:.2f}",
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))  # Use Render's port, default 8000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
