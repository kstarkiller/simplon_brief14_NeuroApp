from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import mlflow
import cv2
import base64
from bson import ObjectId
from datetime import datetime


app = FastAPI()

# Load the ML model
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("runs:/271a79bb8656493ba59699901ab7c2aa/model")


# Define function normalize :
def normalize_image(img, target_size):
    # Convertir en niveaux de gris si ce n'est pas déjà le cas
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    else:
        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Détecter les contours pour trouver le crop optimal
    _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Trouver le contour avec la plus grande aire
        max_contour = max(contours, key=cv2.contourArea)

        # Obtenir les coordonnées du rectangle englobant
        x, y, w, h = cv2.boundingRect(max_contour)

        # Cropper l'image pour obtenir la région d'intérêt
        cropped_img = img[y : y + h, x : x + w]

        # Redimensionner à target_size (pour s'assurer que toutes les images ont la même taille)
        normalized_image = cv2.resize(
            cropped_img, target_size, interpolation=cv2.INTER_AREA
        )
    else:
        # Redimensionner à target_size si aucun contour n'est détecté
        normalized_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return normalized_image


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["braintumor"]


@app.post("/predict/")
async def predict(patient_id: str):

    # Retrieve patient data from MongoDB
    patient_data = db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient_data is None:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # Retrieve scanner_img from patient data
    scanner_img = patient_data.get("scanner_img")
    if scanner_img is None:
        raise HTTPException(
            status_code=400, detail="No scanner image found for the patient."
        )

    # Decode the base64 encoded image data
    image_data = base64.b64decode(scanner_img)
    print(image_data)

    # Convert the bytes to an image
    image_data = np.frombuffer(image_data, np.uint8)
    decoded_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Normalize the image
    image_ready = normalize_image(decoded_image, (224, 224))

    # Convert the image to a list of images (batch of 1)
    image_ready = np.array(image_ready).reshape(1, 224, 224, 3)

    # Make a prediction
    prediction = model.predict(image_ready)

    # Format the prediction
    pred_label = "yes" if prediction[0][0] > 0.5 else "no"
    confidence = float(prediction[0][0])

    # Get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Update patient data with prediction result and date
    db.patients.update_one(
        {"_id": ObjectId(patient_id)},
        {
            "$set": {
                "AI_predict": pred_label,
                "confidence": confidence,
                "prediction_date": current_date,
            }
        },
    )

    # Return the prediction result
    return {
        "AI_predict": pred_label,
        "confidence": confidence,
        "prediction_date": current_date,
    }


# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
