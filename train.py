import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ================== DATASET PATH ==================
dataset_path = "dataset"

# Class labels
labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

data = []
target = []

# ================== LOAD DATA ==================
for label_index, label in enumerate(labels):
    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path, 0)  # grayscale
            img = cv2.resize(img, (128, 128))
            img = img.flatten()

            data.append(img)
            target.append(label_index)

        except:
            continue

# Convert to numpy
data = np.array(data)
target = np.array(target)

print("Dataset Loaded:", data.shape)

# ================== SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# ================== SCALE ==================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================== MODEL ==================
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
model.fit(X_train, y_train)

# ================== EVALUATION ==================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# ================== SAVE MODEL ==================
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# SAVE MODEL
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# SAVE SCALER
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model saved at:", model_path)
print("✅ Scaler saved at:", scaler_path)
