import requests
import cv2
import os
import numpy as np

url = 'http://127.0.0.1:8000/predict/'
yes_path = 'data/raw/yes'
no_path = 'data/raw/no'

# List of images to test
yes_images = [f"{yes_path}/{i}" for i in os.listdir(yes_path)]
no_images = [f"{no_path}/{i}" for i in os.listdir(no_path)]

# Join the lists
all_images = yes_images + no_images

# Choose a random image
filename = np.random.choice(all_images)

# load image using openCV
image = cv2.imread(filename)

# Convert the image to bytes
_, image_encoded = cv2.imencode('.jpg', image)
image_bytes = image_encoded.tobytes()

# Send the image to the API
response = requests.post(url, files={"file": ("image.jpg", image_bytes)})

# Print response managing errors
if response.status_code == 200:
    if response.json().get("label") == "yes":
        print(f'Présence de tumeur ? {response.json().get("label")} (Confidence : {response.json().get("confidence"):.2%})', f'\nFichier envoyé: {filename.split("/")[-1]}')
    else :
        print(f'Présence de tumeur ? {response.json().get("label")} (Confidence : {1/1-response.json().get("confidence"):.2%})', f'\nFichier envoyé: {filename.split("/")[-1]}')
else:
    print("Error:", response.status_code, response.text)
