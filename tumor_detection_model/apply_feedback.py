from hidden import MONGO_URI, MLFLOW_RUN
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import mlflow
import cv2
import base64
from bson import ObjectId
from datetime import dateti


def retrain_model () : 

    # import model 

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.pyfunc.load_model(MLFLOW_RUN)


    #  fit model 
    







    return model