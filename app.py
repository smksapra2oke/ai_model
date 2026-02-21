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

def transform_row(data):
    row = []
    for f in features:
        value = data[f]
        if f in encoders:
            value = encoders[f].transform([value])[0]
        row.append(value)
    return row

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    row = np.array(transform_row(data)).reshape(1, -1)

    pred_index = model.predict(row)[0]
    pred_label = target_encoder.inverse_transform([pred_index])[0]

    probs = model.predict_proba(row)[0]
    class_labels = target_encoder.inverse_transform(range(len(probs)))

    probability = {}
    for label, prob in zip(class_labels, probs):
        probability[label.lower().replace(" ", "_")] = float(prob)

    return jsonify({
        "prediction": pred_label,
        "probability": probability
    })

# 🔥 ENDPOINT BARU
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data_list = request.json  # list of data

    rows = [transform_row(d) for d in data_list]
    rows = np.array(rows)

    preds = model.predict(rows)
    probs_all = model.predict_proba(rows)

    class_labels = target_encoder.inverse_transform(range(probs_all.shape[1]))

    results = []

    for pred_index, probs in zip(preds, probs_all):
        pred_label = target_encoder.inverse_transform([pred_index])[0]

        probability = {}
        for label, prob in zip(class_labels, probs):
            probability[label.lower().replace(" ", "_")] = float(prob)

        results.append({
            "prediction": pred_label,
            "probability": probability
        })

    return jsonify(results)
