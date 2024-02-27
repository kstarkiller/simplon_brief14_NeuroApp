import uvicorn
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Connexion à la base de données MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["braintumor"]  # Remplacez "your_database_name" par le nom de votre base de données MongoDB


# Modèle Pydantic pour les données du patient
class PatientModel(BaseModel):
    name: str
    age: int
    gender: str


# Modèles Pydantic pour la modification du patient
class PatientUpdateModel(BaseModel):
    name: str
    age: int
    gender: str


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

# link with html page
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
    patient_data = patient.dict()
    db.patients.insert_one(patient_data)
    return JSONResponse(content={"redirect_url": "/view_patients"})



# # Route pour visualiser tous les patients avec recherche
# @app.get("/view_patients", response_class=HTMLResponse)
# async def view_patients(request: Request):
#     # Récupérer tous les patients depuis la base de données
#     patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find()]
#     return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})


# Route pour éditer un patient
@app.get("/edit_patient/{patient_id}", response_class=HTMLResponse)
async def edit_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
    return templates.TemplateResponse("edit_patient.html", {"request": request, "patient": patient,
                                                            "patient_id": patient_id})


@app.post("/edit_patient/{patient_id}")
async def edit_patient_post(patient_id: str, patient: PatientUpdateModel):
    # Mettre à jour le patient dans la base de données
    db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": patient.model_dump()})
    return RedirectResponse(url="/view_patients")



# Route for viewing patients with search functionality
@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(
    request: Request,
    search_id: Optional[str] = None, 
    search_name: Optional[str] = None
):
    query_params = {}
    if search_id:
        query_params["_id"] = ObjectId(search_id)
    if search_name:
        query_params["name"] = {"$regex": search_name, "$options": "i"}

    # Retrieve patients based on search criteria
    patients_data = db.patients.find(query_params)
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in patients_data]

    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})

# @app.get("/search_patient", response_class=JSONResponse)
# async def search_patient(patient_id: Optional[str] = None, name: Optional[str] = None):
#     if not patient_id and not name:
#         raise HTTPException(status_code=400, detail="Must provide either patient ID or name for search")
    
#     query = {}
#     if patient_id:
#         try:
#             query["_id"] = ObjectId(patient_id)
#         except:
#             raise HTTPException(status_code=400, detail="Invalid patient ID format")
#     elif name:
#         query["name"] = {"$regex": name, "$options": "i"}
    
#     patients = list(db.patients.find(query))
    
#     if patients:
#         for patient in patients:
#             patient["id"] = str(patient["_id"])
#             del patient["_id"]
#         return patients
#     else:
#         raise HTTPException(status_code=404, detail="Patient not found")

# # Route pour chercher un patient
@app.get("/search_patient", response_class=JSONResponse)
async def search_patient(
    request: Request,
    search_id: Optional[str] = None, 
    search_name: Optional[str] = None
):
    query_params = {}
    if search_id:
        query_params["search_id"] = search_id
    if search_name:
        query_params["search_name"] = search_name
   # Rediriger vers la page "/view_patients" avec les paramètres de recherche
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find()]
    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})




# # Route pour visualiser les patients 
# @app.get("/view_patients", response_class=HTMLResponse)
# async def view_patients(
#     request: Request,
#     search_id: str = Query(None, description="Search by patient ID"),
#     search_name: str = Query(None, description="Search by patient name")
# ):
#     query = {}
#     if search_id:
#         query["_id"] = ObjectId(search_id)
#     if search_name:
#         query["name"] = {"$regex": search_name, "$options": "i"}

#     # Récupérer les patients depuis la base de données selon la recherche
#     patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find(query)]
#     return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})


# # Route pour afficher tous les patients par défaut
# @app.get("/view_patients/all", response_class=HTMLResponse)
# async def view_all_patients(
#     request: Request
# ):
#     # Récupérer tous les patients depuis la base de données
#     patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find()]
#     return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=7070)