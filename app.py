from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ===============================
# LOAD MODEL SEKALI SAJA
# ===============================
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

# ===============================
# HEALTH CHECK
# ===============================
@app.route("/")
def home():
    return "AI Random Forest running"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ===============================
# SAFE TRANSFORM (ANTI ERROR)
# ===============================
def transform_row(data):
    row = []

    for f in features:
        value = data.get(f)

        # Jika field kosong → default 0
        if value is None or value == "":
            row.append(0)
            continue

        # Jika ada encoder
        if f in encoders:
            try:
                # Jika label ada di classes
                if value in encoders[f].classes_:
                    value = encoders[f].transform([value])[0]
                else:
                    # Label baru → pakai 0 sebagai fallback
                    value = 0
            except Exception:
                value = 0

        row.append(value)

    return row


# ===============================
# SINGLE PREDICT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
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

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ===============================
# BATCH PREDICT (SUPER CEPAT)
# ===============================
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        data_list = request.json

        if not isinstance(data_list, list):
            return jsonify({"error": "Input harus berupa list"}), 400

        # Transform semua data
        rows = [transform_row(d) for d in data_list]
        rows = np.array(rows)

        # Predict sekaligus
        preds = model.predict(rows)
        probs_all = model.predict_proba(rows)

        class_labels = target_encoder.inverse_transform(
            range(probs_all.shape[1])
        )

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

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
