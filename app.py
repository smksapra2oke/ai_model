from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

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

@app.route("/")
def home():
    return "AI Random Forest running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    row = []
    for f in features:
        value = data[f]
        if f in encoders:
            value = encoders[f].transform([value])[0]
        row.append(value)

    row = np.array(row).reshape(1, -1)

    # prediksi label
    pred_index = model.predict(row)[0]
    pred_label = target_encoder.inverse_transform([pred_index])[0]

    # probabilitas semua kelas
    probs = model.predict_proba(row)[0]

    class_labels = target_encoder.inverse_transform(range(len(probs)))

    probability = {}
    for label, prob in zip(class_labels, probs):
        key = label.lower().replace(" ", "_")
        probability[key] = float(prob)

    return jsonify({
        "prediction": pred_label,
        "probability": probability
    })
