from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import sqlite3
import json
import pickle
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================== LOAD MODEL ==================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================== LOAD USERS ==================
def load_users():
    with open("users.json", "r") as f:
        return json.load(f)

# ================== DATABASE INIT ==================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        filename TEXT,
        result TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================== MRI VALIDATION ==================
def is_mri_image(filepath):
    img = cv2.imread(filepath)

    if img is None:
        return False, "Invalid image"

    img = cv2.resize(img, (128, 128))
    b, g, r = cv2.split(img)

    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    if diff_rg > 10 or diff_rb > 10 or diff_gb > 10:
        return False, "Only MRI images allowed"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (128 * 128)

    if edge_ratio < 0.01:
        return False, "Invalid MRI structure"

    mean = np.mean(gray)

    if mean < 30 or mean > 200:
        return False, "Invalid MRI intensity"

    return True, "Valid MRI"

# ================== LOGIN ==================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ================== LOGOUT ==================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================== HOME ==================
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ✅ VALIDATION
        valid, message = is_mri_image(filepath)

        if not valid:
            os.remove(filepath)
            return render_template("index.html", error=message, result=None, image=None)

        # ================== PREDICTION ==================
        img = cv2.imread(filepath, 0)
        img = cv2.resize(img, (128, 128))
        img = img.flatten().reshape(1, -1)

        img = scaler.transform(img)
        pred = model.predict(img)[0]
        prob = np.max(model.predict_proba(img)) * 100

        labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        result = labels[pred]

        # ================== SAVE HISTORY ==================
        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("INSERT INTO history (username, filename, result, confidence) VALUES (?, ?, ?, ?)",
                  (session["user"], filename, result, prob))

        conn.commit()
        conn.close()

        return render_template("index.html",
                               result=result,
                               confidence=round(prob, 2),
                               image=filepath)

    return render_template("index.html")

# ================== HISTORY ==================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT filename, result, confidence FROM history WHERE username=?",
              (session["user"],))

    data = c.fetchall()
    conn.close()

    return render_template("history.html", data=data)

# ================== PDF DOWNLOAD ==================
@app.route("/download")
def download():
    if "user" not in session:
        return redirect("/login")

    filename = request.args.get("file")
    result = request.args.get("result")
    confidence = request.args.get("confidence")

    pdf_path = "report.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Alzheimer Prediction Report", styles["Title"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Result: {result}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))
    content.append(Spacer(1, 20))
    content.append(Image(filename, width=200, height=200))

    doc.build(content)

    return send_file(pdf_path, as_attachment=True)

# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True)