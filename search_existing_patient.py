from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["neuroguardlist"]
patients_collection = db["patients"]

class Patient(BaseModel):
    id: int
    name: str
    age: int
    address: str
    brain_scanner: str
    doctor_comment: str

@app.get("/search", response_model=List[Patient])
def get_patients(search_id: Optional[int] = None, search_name: Optional[str] = None):
    query = {}
    if search_id:
        query["id"] = search_id
    if search_name:
        query["name"] = {"$regex": search_name, "$options": "i"}

    patients_data = patients_collection.find(query)
    patients = []
    for patient_data in patients_data:
        patients.append(Patient(**patient_data))

    return patients
