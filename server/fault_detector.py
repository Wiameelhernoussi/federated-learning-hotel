import time
import threading

class FaultDetector:
    def __init__(self, timeout=15):
        self.clients_status = {}  # {client_id: {"last_seen": timestamp, "status": status, "last_round": int, "missed_rounds": int}}
        self.timeout = timeout
        self.lock = threading.Lock()
        
    def update_heartbeat(self, client_id, status, round_num):
        with self.lock:
            if client_id not in self.clients_status:
                self.clients_status[client_id] = {"missed_rounds": 0}
            
            self.clients_status[client_id]["last_seen"] = time.time()
            self.clients_status[client_id]["status"] = status
            self.clients_status[client_id]["last_round"] = round_num

            if status != "INACTIVE":
                self.clients_status[client_id]["missed_rounds"] = 0

    def record_missed_round(self, client_id):
        with self.lock:
            if client_id in self.clients_status:
                self.clients_status[client_id]["missed_rounds"] += 1
                if self.clients_status[client_id]["missed_rounds"] >= 2:
                    self.clients_status[client_id]["status"] = "INACTIVE"
                    print(f"[FAULT DETECTOR] Client {client_id} est marqué INACTIVE (2 rounds manqués).")

    def check_timeouts(self):
        with self.lock:
            current_time = time.time()
            for client_id, info in self.clients_status.items():
                if info.get("status") != "INACTIVE":
                    if current_time - info.get("last_seen", 0) > self.timeout:
                        print(f"[FAULT DETECTOR] Timeout détecté pour le client {client_id} (silencieux depuis {self.timeout}s).")
                        # On ne le marque pas immédiatement INACTIVE, cela dépend des rounds
                        info["status"] = "TIMEOUT"

    def get_status(self):
        with self.lock:
            return self.clients_status.copy()

    def start(self):
        def loop():
            while True:
                self.check_timeouts()
                time.sleep(5)
        t = threading.Thread(target=loop, daemon=True)
        t.start()
