import random
from http.client import HTTPException

from fastapi import FastAPI, HTTPException, Query

from app.db import database, User
import os
import requests
import random
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(title="Verification")

# Directory to store temporarily uploaded images
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

MODEL_PATH = os.path.abspath("model/verification.task")
RECOGNIZER = vision.GestureRecognizer.create_from_model_path(MODEL_PATH)

def recognize_gesture(image_path):
    # Load the input image.
    image = mp.Image.create_from_file(image_path)

    # Run gesture recognition.
    recognition_result = RECOGNIZER.recognize(image)

    # Check if gestures were recognized
    if not recognition_result.gestures:
        raise HTTPException

    # Display the most likely gesture.
    top_gesture = recognition_result.gestures[0][0]
    return top_gesture

async def get_verification(image_url: str = Query(..., description="URL of the image to be verified")):
    # Save the uploaded image temporarily
    image_content = requests.get(image_url).content

    # Save the image temporarily
    timestamp = int(time.time())
    random_number = random.randint(1000, 9999)  # You can adjust the range as needed
    image_id = f"{timestamp}_{random_number}"
    temp_image_path = image_id
    with open(temp_image_path, "wb") as temp_image_file:
        temp_image_file.write(image_content)

    # Perform gesture recognition
    try:
        top_gesture = recognize_gesture(temp_image_path)
    except HTTPException as e:
        return e

    # Remove the temporary image file
    os.remove(temp_image_path)

    return {"top_gesture": top_gesture}