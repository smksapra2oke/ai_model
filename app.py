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

        if value is None or value == "":
            row.append(0)
            continue

        if f in encoders:
            try:
                if value in encoders[f].classes_:
                    value = encoders[f].transform([value])[0]
                else:
                    value = 0
            except:
                value = 0

        row.append(value)

    return row


# ===============================
# ANALISIS PERSONAL (RULE BASED)
# ===============================
def generate_analysis(data, probability, pred_label):
    strengths = []
    risks = []

    nilai_ujikom = float(data.get("nilai_ujikom", 0))
    nilai_kejuruan = float(data.get("nilai_kejuruan", 0))
    pkl = data.get("tempat_pkl_relevan")
    ekskul = data.get("ekskul_aktif")
    pendapatan = float(data.get("pendapatan", 0))

    # ===== NILAI =====
    if nilai_ujikom >= 80:
        strengths.append("Kompetensi teknis sangat baik")
    elif nilai_ujikom >= 70:
        strengths.append("Kompetensi teknis cukup baik")
    else:
        risks.append("Kompetensi teknis perlu ditingkatkan")

    if nilai_kejuruan >= 80:
        strengths.append("Penguasaan keterampilan kejuruan kuat")
    else:
        risks.append("Penguasaan keterampilan kejuruan belum optimal")

    # ===== PKL =====
    if pkl in [1, "1", "Ya", "ya"]:
        strengths.append("Pengalaman PKL relevan dengan bidang kerja")
    else:
        risks.append("Pengalaman PKL kurang relevan")

    # ===== EKSKUL =====
    if ekskul in [1, "1", "Ya", "ya"]:
        strengths.append("Soft skill dan pengalaman organisasi baik")

    # ===== PENDAPATAN =====
    if pendapatan < 2000000:
        risks.append("Pendapatan masih di bawah rata-rata awal karier")

    # ===== SCORE =====
    max_prob = max(probability.values()) if probability else 0
    score = round(max_prob * 100)

    # ===== SUMMARY =====
    summary = (
        f"Berdasarkan analisis model AI, profil alumni berada pada kategori "
        f"{pred_label} dengan tingkat keyakinan {score}%. "
        f"Hasil ini mencerminkan kombinasi antara kompetensi akademik, "
        f"keterampilan praktis, dan kesiapan kerja."
    )

    return {
        "score": score,
        "strengths": strengths,
        "risks": risks,
        "summary": summary
    }


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

        # 🔎 ANALISIS PERSONAL
        analysis = generate_analysis(data, probability, pred_label)

        return jsonify({
            "prediction": pred_label,
            "probability": probability,
            "analysis": analysis
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ===============================
# BATCH PREDICT
# ===============================
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        data_list = request.json

        if not isinstance(data_list, list):
            return jsonify({"error": "Input harus berupa list"}), 400

        rows = [transform_row(d) for d in data_list]
        rows = np.array(rows)

        preds = model.predict(rows)
        probs_all = model.predict_proba(rows)

        class_labels = target_encoder.inverse_transform(
            range(probs_all.shape[1])
        )

        results = []

        for data, pred_index, probs in zip(data_list, preds, probs_all):
            pred_label = target_encoder.inverse_transform([pred_index])[0]

            probability = {}
            for label, prob in zip(class_labels, probs):
                probability[label.lower().replace(" ", "_")] = float(prob)

            analysis = generate_analysis(data, probability, pred_label)

            results.append({
                "prediction": pred_label,
                "probability": probability,
                "analysis": analysis
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
