# pyrefly: ignore [missing-import]
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

RUN_MODE = os.environ.get("RUN_MODE", "iid")

app = Flask(__name__)
CORS(app)

# Variables globales que le server.py mettra à jour
global_state = {
    "current_round": 0,
    "metrics": {
        "loss": 0.0,
        "accuracy": 0.0,
        "duration": 0.0,
        "byzantine_excluded": 0,
        "lamport_time": 0,
        "messages_lost": 0,
        "message_loss_rate": float(os.environ.get("MESSAGE_LOSS_RATE", "0.0"))
    },
    "fault_detector": None  # Référence vers l'instance de FaultDetector
}

@app.route('/metrics', methods=['GET'])
def get_metrics():
    m = global_state["metrics"]
    return jsonify({
        "round":             global_state["current_round"],
        "loss":              m["loss"],
        "accuracy":          m["accuracy"],
        "duration":          m.get("duration", 0.0),
        "byzantine_excluded":m.get("byzantine_excluded", 0),
        "lamport_time":      m.get("lamport_time", 0),
        "messages_lost":     m.get("messages_lost", 0),
        "message_loss_rate": m.get("message_loss_rate", 0.0)
    })

@app.route('/status', methods=['GET'])
def get_status():
    if global_state["fault_detector"]:
        clients = global_state["fault_detector"].get_status()
        clients_list = [
            {"id": cid, "statut": info.get("status", "UNKNOWN"), "last_round": info.get("last_round", 0)}
            for cid, info in clients.items()
        ]
        return jsonify({"clients": clients_list})
    return jsonify({"clients": []})

@app.route('/history', methods=['GET'])
def get_history():
    if os.path.exists("metrics.csv"):
        try:
            df = pd.read_csv("metrics.csv")
            # Ensure mode column exists for backward compatibility
            if "mode" not in df.columns:
                df["mode"] = "iid"
            history = df.to_dict(orient='records')
            return jsonify(history)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])

@app.route('/checkpoint', methods=['GET'])
def get_checkpoint():
    if os.path.exists("round_state.json"):
        try:
            import json
            with open("round_state.json") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"status": "no checkpoint"})

def run_api(host='0.0.0.0', port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)
