from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import mlflow
import mlflow.keras
import cv2
import sys
sys.path.append('../')
from functions.normalize_images import normalize_image

# Load the model
mlflow.set_tracking_uri('http://localhost:5000')
model = mlflow.pyfunc.load_model('runs:/cd2378db4d16452a831d248eea8811a7/models')

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Check if the file is an image
    if file is None:
        return JSONResponse(status_code=400, content={"message": "No file uploaded."})
    
    # Read the image
    image_data = await file.read()
    # convert the bytes to image
    image_data = np.frombuffer(image_data, np.uint8)
    # decode the bytes to image
    decoded_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Normalize the image
    image_ready = normalize_image(decoded_image, (224, 224))
    # Reshape and convert the image to a list of images (batch of 1)
    image_ready = np.array(image_ready).reshape(1, 224, 224, 3)

    # Make a prediction
    prediction = model.predict(image_ready)
    # Format the prediction
    pred_label = "yes" if prediction[0][0] > 0.5 else "no"

    return {"label": pred_label, "confidence": float(prediction[0][0])}

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)