import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ✅ Allow only image files
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Load model and labels
model = load_model("dogbreed.h5")

with open("labels.json", "r") as f:
    labels = json.load(f)

IMG_SIZE = (224, 224)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # ✅ POST: handle uploaded image
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    # ✅ File type validation
    if not allowed_file(file.filename):
        return "Only JPG/PNG images are allowed", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # ✅ Load and preprocess image
    img = image.load_img(file_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    predicted_breed = labels[str(top_idx)]

    return render_template(
        "output.html",
        prediction=predicted_breed,
        confidence=f"{confidence:.2f}%"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
