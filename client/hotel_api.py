# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timezone
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Shared state — written by client.py, read by this API
# ---------------------------------------------------------------------------
shared_state = {
    "model": None,
    "current_round": 0,
    "accuracy": 0.0,
    "loss": 0.0,
    "hotel_name": "",
    "client_id": 0,
    "status": "STARTING",
    "predictions_history": []
}

# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------
MEAL_MAP = {"BB": 0.2, "HB": 0.5, "FB": 0.8, "SC": 0.0}
SEGMENT_MAP = {
    "Direct": 0.1, "Corporate": 0.3, "Online TA": 0.6,
    "Offline TA/TO": 0.8, "Groups": 0.5
}

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def booking_to_tensor(data):
    """
    Map the 11 hotel-booking fields to a (1,1,28,28) normalised tensor.
    Each of the 784 positions gets the value of field[i % num_fields].
    """
    lead_time          = _clamp(float(data.get("lead_time", 0)),            0, 365) / 365
    stays_weekend      = _clamp(float(data.get("stays_weekend", 0)),        0, 7)   / 7
    stays_week         = _clamp(float(data.get("stays_week", 0)),           0, 14)  / 14
    adults             = _clamp(float(data.get("adults", 1)) - 1,           0, 3)   / 3
    meal               = MEAL_MAP.get(str(data.get("meal", "BB")), 0.2)
    market_segment     = SEGMENT_MAP.get(str(data.get("market_segment", "Direct")), 0.4)
    is_repeated_guest  = float(bool(data.get("is_repeated_guest", 0)))
    prev_cancellations = _clamp(float(data.get("previous_cancellations", 0)), 0, 10) / 10
    booking_changes    = _clamp(float(data.get("booking_changes", 0)),       0, 10)  / 10
    adr                = _clamp(float(data.get("adr", 100)),                 0, 500) / 500
    special_requests   = _clamp(float(data.get("special_requests", 0)),      0, 5)   / 5

    fields = [
        lead_time, stays_weekend, stays_week, adults,
        meal, market_segment, is_repeated_guest,
        prev_cancellations, booking_changes, adr, special_requests
    ]
    num_fields = len(fields)

    pixels = np.array([fields[i % num_fields] for i in range(784)], dtype=np.float32)
    tensor = torch.tensor(pixels).reshape(1, 1, 28, 28)
    # MNIST normalisation
    tensor = (tensor - 0.1307) / 0.3081
    return tensor

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def serve_dashboard():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, "hotel_dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    model = shared_state["model"]
    if model is None:
        return jsonify({"error": "Model not ready yet — please wait for the first FL round to complete."}), 503

    data = request.get_json(force=True, silent=True) or {}

    try:
        tensor = booking_to_tensor(data)
    except Exception as e:
        return jsonify({"error": f"Input error: {e}"}), 400

    try:
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().numpy()
    except Exception as e:
        return jsonify({"error": f"Inference error: {e}"}), 500

    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    # Classes 5-9 = high cancellation risk
    cancellation_prob = float(np.sum(probs[5:]))
    cancellation_prob = round(cancellation_prob, 4)

    if cancellation_prob < 0.4:
        risk_level = "LOW"
        risk_emoji = "🟢"
    elif cancellation_prob < 0.7:
        risk_level = "MEDIUM"
        risk_emoji = "🟡"
    else:
        risk_level = "HIGH"
        risk_emoji = "🔴"

    result = {
        "hotel": shared_state["hotel_name"],
        "cancellation_probability": cancellation_prob,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "model_round": shared_state["current_round"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    history = shared_state["predictions_history"]
    history.append({**result, **{k: data.get(k) for k in data}})
    shared_state["predictions_history"] = history[-50:]

    return jsonify(result)


@app.route("/client-status", methods=["GET"])
def client_status():
    hist = shared_state["predictions_history"]
    return jsonify({
        "hotel": shared_state["hotel_name"],
        "client_id": shared_state["client_id"],
        "status": shared_state["status"],
        "current_round": shared_state["current_round"],
        "accuracy": shared_state["accuracy"],
        "loss": shared_state["loss"],
        "total_predictions": len(hist),
        "last_prediction": hist[-1] if hist else None
    })


@app.route("/predictions-history", methods=["GET"])
def predictions_history():
    return jsonify(shared_state["predictions_history"][-20:])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "hotel": shared_state["hotel_name"]})


# ---------------------------------------------------------------------------
# Entry point called from client.py
# ---------------------------------------------------------------------------
def run_hotel_api(host="0.0.0.0", port=8000):
    app.run(host=host, port=port, debug=False, use_reloader=False)
