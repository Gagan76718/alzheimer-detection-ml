import cv2
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

classes = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]

def predict_image(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten()

    img = np.array([img])
    img = scaler.transform(img)

    probs = model.predict_proba(img)[0]
    pred = np.argmax(probs)
    confidence = round(max(probs) * 100, 2)

    return classes[pred], confidence, probs