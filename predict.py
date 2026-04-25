import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle

# ================= SETTINGS =================
IMG_SIZE = 64
LATENT_DIM = 128

labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# ================= AUTOENCODER =================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, LATENT_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ================= LOAD MODELS =================
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
autoencoder.eval()

model = pickle.load(open("classifier.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# ================= PREDICT FUNCTION =================
def predict_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return "Invalid Image", 0.0

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.flatten().astype(np.float32)

    img_tensor = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        encoded = autoencoder.encoder(img_tensor).numpy()

    encoded = scaler.transform(encoded)

    pred = model.predict(encoded)[0]
    confidence = float(np.max(model.predict_proba(encoded)))

    return labels[pred], round(confidence * 100, 2)