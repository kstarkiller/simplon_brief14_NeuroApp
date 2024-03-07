# simplon_brief14_NeuroApp

## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [Backend Documentation](#backend-documentation)
   1. [Key Features](#key-features-backend)
   2. [Pydantic Models](#pydantic-models)
   3. [Data Validation](#data-validation)
   4. [Endpoints](#endpoints-backend)
      1. [Home and Patient Management](#home-and-patient-management)
      2. [Patient Data Handling](#patient-data-handling)
      3. [AI Prediction and Validation](#ai-prediction-and-validation)
      4. [Feedback and Error Reporting](#feedback-and-error-reporting)
      5. [Additional Functions](#additional-functions)
   5. [Running the backend](#running-the-backend)
4. [Brain Tumor Prediction API Documentation](#brain-tumor-prediction-api-documentation)
   1. [Key Features](#key-features-api)
   2. [Core Functionalities](#core-functionalities)
      1. [ML Model Loading](#ml-model-loading)
      2. [Image Processing](#image-processing)
      3. [Database Connection](#database-connection)
   3. [Endpoints](#endpoints-api)
      1. [Prediction](#prediction)
   4. [Running the model's api](#running-the-model-api)

## Description

The NeuroApp is an application designed to list, edit, and predict tumors in patients. It provides a user-friendly interface for medical professionals to manage potential tumor patients. With the NeuroApp, users can easily input and update patient information, view and edit tumor details, and utilize predictive algorithms to predict tumor's presence.

## Installation

To install the requirements, follow these steps:

1. Clone the repository: `git clone https://github.com/kstarkiller/simplon_brief14_NeuroApp.git`
2. Install the requirements using `pip install -r requirements.txt`

## Backend Documentation

This backend is implemented using FastAPI and integrates with MongoDB for data management. It serves various endpoints for handling patient data and performing AI-based predictions for brain tumor diagnosis.

### Key Features

- Patient data management (add, update, view, search)
- Image-based AI prediction for brain tumor detection
- Real-time feedback and validation of predictions

### Pydantic Models

- **PredictionModel**: Manages AI predictions.
- **ScannerModel**: Handles scanner image data.
- **PatientModel**: Represents a patient's data.
- **PatientUpdateModel**: Used for updating patient data.
- **PatientViewModel**: Used for viewing patient data.

### Data Validation

- Validations, such as rounding off confidence scores in `PredictionModel`, are implemented using Pydantic validators.

### Endpoints

#### Home and Patient Management

- `GET /`: The home page.
- `GET /add_patient`: Form to add a new patient.
- `POST /add_patient`: Endpoint to submit new patient data.
- `GET /view_patients`: View all patients with optional filtering.

#### Patient Data Handling

- `GET /full_view_patient/{patient_id}`: Detailed view of a specific patient.
- `GET /edit_patient/{patient_id}`: Form to edit patient data.
- `POST /edit_patient/{patient_id}`: Submit updated patient data.
- `GET /search_patient`: Search for patients by ID or name.

#### AI Prediction and Validation

- `GET /predict_patient/{patient_id}`: Trigger AI prediction for a patient.
- `GET /check_predict`: Interface to check prediction results.
- `POST /check_predict_post/{patient_id}`: Submit prediction validation.

#### Feedback and Error Reporting

- `GET /feed_back`: View feedback and error reports.

#### Additional Functions

- `trigger_prediction(image_data)`: Function to trigger AI prediction requests to a model API.

### Running the backend

- Configured to run locally, accessible via port `3000`.
- `cd braintumor-ui`
- run `python app.py`

## Brain Tumor Prediction API Documentation

This API, built with FastAPI, integrates machine learning (ML) for brain tumor predictions and connects to a MongoDB database for handling patient data. It's designed to predict brain tumors using scanned images.

### Key Features

- Integration with MLflow for model management.
- Image processing for preparing data for ML predictions.
- Real-time prediction on patient scanned images.

### Core Functionalities

#### ML Model Loading

- Utilizes MLflow to load the pre-trained model, which is used for making predictions.

#### Image Processing

- `normalize_image(img, target_size)`: Function to process and normalize images, including grayscale conversion, denoising, contour detection, and resizing.

#### Database Connection

- Establishes a connection to MongoDB for accessing and storing patient data.

### Endpoints

#### Prediction

- `POST /predict/`: Receives a patient ID and returns the AI model's prediction. It involves:
  - Fetching patient data from MongoDB.
  - Image decoding and processing for the prediction.
  - Applying the ML model to predict the presence of a tumor.
  - Returning the prediction result with confidence levels.

### Running the model api

- Configured to run locally, accessible via port `8000`.
- `cd api`
- run `python model_api.py`
