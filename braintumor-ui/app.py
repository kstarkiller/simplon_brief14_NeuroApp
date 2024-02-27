import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hidden import MONGO_URI

import uvicorn
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional
import base64
import binascii

app = FastAPI()

# Connexion à la base de données MongoDB
client = MongoClient(MONGO_URI)
db = client["braintumor"]


# Modèles Pydantic pour l'ajout d'un patient
class PatientModel(BaseModel):
    name: str = Form(...)
    age: int = Form(...)
    gender: str = Form(...)
    scanner_img: Optional[str] = Form(None, description="Base64 encoded image")
    scanner_name: Optional[str] = Form(None)

    @property
    def image_bytes(self):
        if self.scanner_img is not None:
            try:
                return base64.b64decode(self.scanner_img)
            except binascii.Error:
                return None
        return None


# Modèles Pydantic pour la modification du patient
class PatientUpdateModel(BaseModel):
    name: Optional[str] = Form(None)
    age: Optional[int] = Form(None)
    gender: Optional[str] = Form(None)
    scanner_img: Optional[str] = Form(None, description="Base64 encoded image")
    scanner_name: Optional[str] = Form(None)

    @property
    def image_bytes(self):
        if self.scanner_img is not None:
            try:
                return base64.b64decode(self.scanner_img)
            except binascii.Error:
                return None
        return None


# Modèles Pydantic pour la visualisation des patients
class PatientViewModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str


# Modèle Pydantic pour les prédictions (à adapter selon vos besoins)
class PredictionModel(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    pass


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
    db.patients.insert_one(patient_data)
    return JSONResponse(content={"redirect_url": "/view_patients"})


# Route pour visualiser tous les patients
@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(request: Request, name: Optional[str] = None, patient_id: Optional[str] = None):
    # Récupérer tous les patients depuis la base de données
    query = {}
    if name:
        query["name"] = {"$regex": name, "$options": "i"}
    if patient_id:
        try:
            query["_id"] = ObjectId(patient_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
    
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find(query)]
    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})

# Route pour éditer un patient
@app.get("/edit_patient/{patient_id}", response_class=HTMLResponse)
async def edit_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient_data = db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient_data is not None:
        patient = PatientModel(**{str(k): v for k, v in patient_data.items()})
        return templates.TemplateResponse("edit_patient.html", {"request": request, "patient": patient,
                                                                "patient_id": patient_id})
    else:
        return JSONResponse(content={"error": "Patient not found"})


@app.post("/edit_patient/{patient_id}")
async def edit_patient_post(patient_id: str, patient: PatientUpdateModel):
    # Obtenir un dictionnaire des champs définis
    updated_fields = {k: v for k, v in patient.model_dump().items() if v is not None}

    # Mettre à jour uniquement les champs définis dans la base de données
    db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": updated_fields})

    return RedirectResponse(url="/view_patients")


@app.get("/search_patient", response_class=JSONResponse)
async def search_patient(patient_id: Optional[str] = None, name: Optional[str] = None):
    if not patient_id and not name:
        raise HTTPException(status_code=400, detail="Must provide either patient ID or name for search")
    
    query = {}
    if patient_id:
        try:
            query["_id"] = ObjectId(patient_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
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

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=3000)
