from datetime import datetime
import requests
import binascii
import base64
from typing import Optional
from pydantic import BaseModel, validator
from bson import ObjectId
from pymongo import MongoClient
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form, HTTPException
import uvicorn
import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))
from hidden import MONGO_URI, MLFLOW_RUN, DEV_MODE


app = FastAPI()

# Connexion à la base de données MongoDB
client = MongoClient(MONGO_URI)
db = client["braintumor"]

# Modèle Pydantic pour les prédictions (à adapter selon vos besoins)


class PredictionModel(BaseModel):
    AI_predict: Optional[str] = None
    confidence: Optional[float] = None
    prediction_date: Optional[str] = None
    predict_check: Optional[str] = None
    predict_check_date: Optional[str] = None
    comment: Optional[str] = None

    @validator('confidence')
    def validate_confidence(cls, v):
        return round(v, 2) if v is not None else v

    def model_dump(self):
        return self.__dict__

# Modèle Pydantic pour le scanner


class ScannerModel(BaseModel):
    scanner_img: Optional[str] = Form(None, description="Base64 encoded image")
    scanner_name: Optional[str] = Form(None)
    prediction: Optional[PredictionModel] = None

    @property
    def image_bytes(self):
        if self.scanner_img is not None:
            try:
                return base64.b64decode(self.scanner_img)
            except binascii.Error:
                return None
        return None

# Modèles Pydantic pour l'ajout d'un patient


class PatientModel(BaseModel):
    name: str = Form(...)
    age: int = Form(...)
    gender: str = Form(...)
    scanner: Optional[ScannerModel] = None

    def model_dump(self):
        return self.__dict__

# Modèles Pydantic pour la modification du patient


class PatientUpdateModel(BaseModel):
    name: Optional[str] = Form(None)
    age: Optional[int] = Form(None)
    gender: Optional[str] = Form(None)
    scanner: Optional[ScannerModel] = None

    def model_dump(self):
        return self.__dict__

# Modèles Pydantic pour la visualisation des patients


class PatientViewModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str
    scanner: Optional[ScannerModel] = None


# Montez le répertoire 'static' pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")


# Route pour la page d'accueil
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route pour ajouter un patient
@app.get("/add_patient", response_class=HTMLResponse)
def add_patient(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})


@app.post("/add_patient")
async def add_patient_post(patient: PatientModel):
    # Insérer le patient dans la base de données
    patient_data = patient.model_dump()
    if patient_data.get('scanner'):
        patient_data['scanner'] = patient_data['scanner'].dict()
    db.patients.insert_one(patient_data)
    return JSONResponse(content={"redirect_url": "/view_patients"})

# endpoint full_view_patient


@app.get("/full_view_patient/{patient_id}", response_class=HTMLResponse)
async def full_view_patient(request: Request, patient_id: str):
    # Retrieve patient information from the database
    patient_data = db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient_data is not None:
        # Prepare the data to pass to the HTML template
        patient = PatientViewModel(id=str(patient_data["_id"]), **patient_data)
        return templates.TemplateResponse(
            "full_view_patient.html",
            {"request": request, "patient": patient, "patient_id": patient_id}
        )
    else:
        raise HTTPException(status_code=404, detail="Patient not found")

# Route pour visualiser tous les patients


@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(
        request: Request,
        name: Optional[str] = None,
        patient_id: Optional[str] = None):
    # Récupérer tous les patients depuis la base de données
    query = {}
    if name:
        query["name"] = {"$regex": name, "$options": "i"}
    if patient_id:
        try:
            query["_id"] = ObjectId(patient_id)
        except:
            raise HTTPException(
                status_code=400, detail="Invalid patient ID format")

    patients = [
        PatientViewModel(id=str(patient["_id"]), **patient)
        for patient in db.patients.find(query)
    ]
    return templates.TemplateResponse(
        "view_patients.html", {"request": request, "patients": patients}
    )

# Route pour visualiser tous les patients validés


@app.get("/view_validates_patients", response_class=HTMLResponse)
async def view_validates_patients(request: Request):

    query = {}

    patients = [
        PatientViewModel(id=str(patient["_id"]), **patient)
        for patient in db.patients.find(query)
    ]
    return templates.TemplateResponse(
        "view_validates_patients.html", {
            "request": request, "patients": patients}
    )


# Route pour visualiser tous les patients en attente de validation
@app.get("/view_waiting_patients", response_class=HTMLResponse)
async def view_waiting_patients(request: Request):

    query = {}

    patients = [
        PatientViewModel(id=str(patient["_id"]), **patient)
        for patient in db.patients.find(query).sort("scanner.prediction.confidence", -1)
    ]
    return templates.TemplateResponse(
        "view_waiting_patients.html", {
            "request": request, "patients": patients}
    )


# Route pour éditer un patient
@app.get("/edit_patient/{patient_id}", response_class=HTMLResponse)
async def edit_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient_data = db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient_data is not None:
        patient = PatientModel(**{str(k): v for k, v in patient_data.items()})
        return templates.TemplateResponse(
            "edit_patient.html",
            {"request": request, "patient": patient, "patient_id": patient_id},
        )
    else:
        return JSONResponse(content={"error": "Patient not found"})


# To update mongoDB with new datas edited
@app.post("/edit_patient/{patient_id}")
async def edit_patient_post(patient_id: str, patient: PatientUpdateModel):
    # Obtenir un dictionnaire des champs définis
    updated_fields = {k: v for k,
                      v in patient.model_dump().items() if v is not None}
    if updated_fields.get('scanner'):
        scanner_fields = updated_fields['scanner'].dict()
        for key, value in scanner_fields.items():
            updated_fields[f'scanner.{key}'] = value
        del updated_fields['scanner']

    # Mettre à jour uniquement les champs définis dans la base de données
    db.patients.update_one({"_id": ObjectId(patient_id)}, {
                           "$set": updated_fields})

    return RedirectResponse(url="/view_patients")


@app.get("/search_patient", response_class=JSONResponse)
async def search_patient(patient_id: Optional[str] = None, name: Optional[str] = None):
    if not patient_id and not name:
        raise HTTPException(
            status_code=400, detail="Must provide either patient ID or name for search"
        )
    query = {}
    if patient_id:
        try:
            query["_id"] = ObjectId(patient_id)
        except:
            raise HTTPException(
                status_code=400, detail="Invalid patient ID format")
    elif name:
        query["name"] = {"$regex": name, "$options": "i"}

    patients = list(db.patients.find(query))

    if patients:
        for patient in patients:
            patient["id"] = str(patient["_id"])
            del patient["_id"]
        return patients
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


# Route pour faire la prediction
@app.get("/predict_patient/{patient_id}", response_class=HTMLResponse)
async def predict_patient(request: Request, patient_id: str):
    url = f"http://localhost:8000/predict/?patient_id={patient_id}"
    prediction_result = requests.post(url)
    if prediction_result.status_code == 200:
        prediction_result = prediction_result.json()
        if prediction_result:
            patient_data = db.patients.find_one({"_id": ObjectId(patient_id)})
            if patient_data.get("scanner") and patient_data["scanner"].get("prediction") is None:
                db.patients.update_one(
                    {"_id": ObjectId(patient_id)},
                    {"$set": {"scanner.prediction": {}}}
                )
            db.patients.update_one(
                {"_id": ObjectId(patient_id)},
                {"$set": {
                    "scanner.prediction.AI_predict": 'Tumor' if prediction_result["AI_predict"] == "yes" else 'No tumor',
                    "scanner.prediction.confidence": (1 - prediction_result["confidence"])*100 if prediction_result["AI_predict"] == "no" else prediction_result["confidence"]*100,
                    "scanner.prediction.prediction_date": prediction_result["prediction_date"]
                }}
            )
            return HTMLResponse(
                content=f"""
                <html>
                    <head>
                        <title>Prediction Success</title>
                        <meta http-equiv='refresh' content='3;url=/full_view_patient/{patient_id}' />
                    </head>
                    <body style="font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                        <div style="text-align: center;">
                            <h1>Prediction Successful</h1>
                            <p>The prediction has been processed successfully.</p>
                            <p>Redirecting to patient's full view...</p>
                        </div>
                    </body>
                </html>
                """
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Prediction failed. Please check if the image exists.",
            )
    else:
        raise HTTPException(
            status_code=500, detail="Prediction request failed.")

# Route pour faire le check de la prediction


@app.get("/check_predict", response_class=HTMLResponse)
def check_predict(request: Request):
    return templates.TemplateResponse("view_full_patient.html", {"request": request})

# Route pour faire le check de la prediction


@app.post("/check_predict_post/{patient_id}")
async def check_predict_post(patient_id: str, patient: PredictionModel):
    try:
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Update patient data with check result and date
        predict_check = patient.model_dump().get("predict_check")
        comment = patient.model_dump().get("comment")
        # Update patient data with check result and date
        db.patients.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": {
                "scanner.prediction.predict_check": predict_check,
                "scanner.prediction.predict_check_date": current_date
            }}
        )
        # If 'no' is selected, also update the comment
        if predict_check == "no":
            db.patients.update_one(
                {"_id": ObjectId(patient_id)},
                {"$set": {
                    "scanner.prediction.comment": comment
                }}
            )

        # Return the page
        return RedirectResponse(url="/view_patients")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def trigger_prediction(image_data: str):
    # Trigger prediction request to model API
    model_api_url = "http://localhost:8000/predict/"
    files = {"file": ("image.jpg", image_data)}
    try:
        response = requests.post(model_api_url, files=files)
        if response.status_code == 200:
            prediction_result = response.json()
            return prediction_result
        else:
            return None
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

# Route pour voir le feedback des erreurs


@app.post("/feed_back")
async def feed_back(request: Request):
    data = await request.json()
    url = "http://localhost:8000/feedback/"

    requests.post(url, json=data)

    # return templates.TemplateResponse(
    #     "view_validates_patients.html", {"request": request}
    # )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)
