from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # urutan fitur HARUS sama
    features = [
        "jurusan",
        "nilai_ujikom",
        "nilai_kejuruan",
        "tempat_pkl_relevan",
        "ekskul_aktif",
        "status_tracer",
        "bidang_pekerjaan",
        "jabatan_pekerjaan",
        "pendapatan"
    ]

    row = []
    for f in features:
        value = data[f]
        if f in encoders:
            value = encoders[f].transform([value])[0]
        row.append(value)

    pred = model.predict([row])[0]
    label = target_encoder.inverse_transform([pred])[0]

    return jsonify({"prediction": label})