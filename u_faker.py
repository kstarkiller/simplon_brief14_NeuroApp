from pymongo import MongoClient
from faker import Faker
import random

fake = Faker()

client = MongoClient("mongodb://localhost:27017/")
db = client["braintumor"] 
def create_fake_patient():
    return {
        "name": fake.name(),
        "age": random.randint(1, 100),
        "gender": random.choice(["male", "female"])
    }

def insert_fake_patients(n):
    fake_patients = [create_fake_patient() for _ in range(n)]
    db.patients.insert_many(fake_patients)
    print(f"{n} fake patients inserted into the database.")

if __name__ == "__main__":
    insert_fake_patients(40)
    