from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import json
from predict import predict_image

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"


# ================= USER FILE =================
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


# ================= HISTORY FILE =================
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []


def save_history(data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")


# ================= SIGNUP =================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        if username in users:
            flash("User already exists")
        else:
            users[username] = {"password": password}
            save_users(users)
            flash("Signup successful")
            return redirect(url_for("login"))

    return render_template("signup.html")


# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result, confidence = predict_image(filepath)
            confidence = float(confidence)  # 🔥 FIX JSON ERROR

            image_path = filepath

            # SAVE HISTORY
            history = load_history()
            history.append({
                "user": session["user"],
                "image": filepath,
                "result": result,
                "confidence": confidence
            })
            save_history(history)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )


# ================= HISTORY =================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    history = load_history()
    user_history = [h for h in history if h["user"] == session["user"]]

    return render_template("history.html", history=user_history)


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)