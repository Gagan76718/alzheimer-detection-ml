import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ================= SETTINGS =================
IMG_SIZE = 64
LATENT_DIM = 128
EPOCHS = 80
BATCH_SIZE = 64

dataset_path = "dataset"
labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# ================= LOAD DATA =================
data = []
target = []

print("Loading dataset...")

for i, label in enumerate(labels):
    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        data.append(img.flatten())
        target.append(i)

data = np.array(data, dtype=np.float32)
target = np.array(target)

print("Dataset Loaded:", data.shape)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, stratify=target, random_state=42
)

X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)

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


autoencoder = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

print("Training Autoencoder...")

for epoch in range(EPOCHS):
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch = X_train_tensor[indices]

        optimizer.zero_grad()
        encoded, decoded = autoencoder(batch)
        loss = criterion(decoded, batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# ================= FEATURE EXTRACTION =================
with torch.no_grad():
    X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()
    X_test_encoded = autoencoder.encoder(X_test_tensor).numpy()

# ================= SCALING =================
scaler = StandardScaler()
X_train_encoded = scaler.fit_transform(X_train_encoded)
X_test_encoded = scaler.transform(X_test_encoded)

# ================= ANN =================
model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=800,
    learning_rate_init=0.0005,
    early_stopping=True,
    verbose=True
)

print("Training ANN...")
model.fit(X_train_encoded, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🔥 FINAL ACCURACY: {accuracy*100:.2f}%")

# ================= SAVE =================
torch.save(autoencoder.state_dict(), "autoencoder.pth")
pickle.dump(model, open("classifier.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Models saved successfully!")