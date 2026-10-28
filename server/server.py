import grpc
from concurrent import futures
import time
import os
import json
import random
import torch
import torch.nn as nn
import threading
import sys

# Import generated gRPC files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
import fl_pb2
import fl_pb2_grpc

from aggregator import fedavg, serialize_weights, deserialize_weights, log_metrics, byzantine_filter
from fault_detector import FaultDetector
from lamport_clock import LamportClock
import api

run_mode = os.environ.get("RUN_MODE", "iid")

# ── Network message-loss simulator ────────────────────────────────────────────
class NetworkSimulator:
    """
    Simule aléatoirement des pertes de messages gRPC.
    Activer avec la variable d'env MESSAGE_LOSS_RATE (float 0.0–0.3).
    """
    def __init__(self):
        rate = float(os.environ.get("MESSAGE_LOSS_RATE", "0.0"))
        self.loss_rate = min(0.3, max(0.0, rate))   # Clamp to [0, 30%]
        self.messages_lost = 0
        self.lock = threading.Lock()

    def should_drop(self, label="message", client_id="?"):
        """Retourne True si le message doit être ignoré."""
        if self.loss_rate == 0.0:
            return False
        if random.random() < self.loss_rate:
            with self.lock:
                self.messages_lost += 1
            print(f"[NETWORK SIM] {label} du client {client_id} perdu "
                  f"(taux={self.loss_rate*100:.0f}%)")
            return True
        return False

    def get_count(self):
        with self.lock:
            return self.messages_lost

# Model architecture to initialize global weights
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class FederatedServer(fl_pb2_grpc.FederatedServiceServicer):
    def __init__(self):
        self.num_rounds = int(os.environ.get("NUM_ROUNDS", 20))
        self.min_fit_clients = int(os.environ.get("MIN_FIT_CLIENTS", 2))
        self.round_timeout = int(os.environ.get("ROUND_TIMEOUT", 30))
        
        self.current_round = 1
        self.clients = set()
        self.fault_detector = FaultDetector(timeout=15)
        self.fault_detector.start()

        # Lamport logical clock
        self.clock = LamportClock()

        # Network loss simulator
        self.net_sim = NetworkSimulator()
        print(f"[SERVER] NetworkSimulator: taux de perte = {self.net_sim.loss_rate*100:.0f}%")

        # Link to Flask API
        api.global_state["fault_detector"] = self.fault_detector
        
        # Initial global model
        self.global_weights = self.init_global_weights()
        self.load_model()
        self.load_round_checkpoint()
        
        # Round state
        self.round_lock = threading.Lock()
        self.received_updates = []
        self.received_sizes = []
        self.received_losses = []
        self.received_accuracies = []
        self.aggregating = False
        self._round_start_time = time.time()
        
    def init_global_weights(self):
        model = SimpleCNN()
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
        
    def load_model(self):
        # Look for the highest round model
        models = [f for f in os.listdir('.') if f.startswith('model_round_') and f.endswith('.pt')]
        if not models:
            return
        rounds = [int(f.split('_')[2].split('.')[0]) for f in models]
        # Try from the most recent round downward until we find a valid checkpoint
        for max_round in sorted(rounds, reverse=True):
            model_path = f"model_round_{max_round}.pt"
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                model = SimpleCNN()
                model.load_state_dict(state_dict)
                self.global_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
                self.current_round = max_round + 1
                print(f"[SERVER] Modèle chargé depuis {model_path}, reprise au round {self.current_round}")
                api.global_state["current_round"] = self.current_round - 1
                return
            except Exception as e:
                print(f"[SERVER] Fichier corrompu {model_path} ({e}), suppression et tentative sur le round précédent...")
                try:
                    os.remove(model_path)
                except OSError:
                    pass
        print("[SERVER] Aucun checkpoint valide trouvé, démarrage depuis zéro.")

    def save_model(self, round_num):
        model = SimpleCNN()
        state_dict = model.state_dict()
        for key, weight in zip(state_dict.keys(), self.global_weights):
            state_dict[key] = torch.tensor(weight)
        model.load_state_dict(state_dict)
        # Écriture atomique : temp file + rename pour éviter les fichiers corrompus
        tmp_path = f"model_round_{round_num}.pt.tmp"
        final_path = f"model_round_{round_num}.pt"
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, final_path)
        print(f"[SERVER] Checkpoint sauvegardé : {final_path}")

    def Register(self, request, context):
        client_id = request.client_id
        t = self.clock.tick()
        with self.round_lock:
            self.clients.add(client_id)
            self.fault_detector.update_heartbeat(client_id, "READY", self.current_round)
        print(f"[SERVER] Client {client_id} enregistré.")
        print(f"[LAMPORT] Client {client_id} enregistré à t={t}")
        return fl_pb2.RegistrationAck(success=True, message="Enregistré avec succès.")

    def save_round_checkpoint(self):
        """Persist current round progress so crashes can be diagnosed."""
        state = {
            "current_round": self.current_round,
            "num_received": len(self.received_updates),
            "timestamp": time.time()
        }
        try:
            with open("round_state.json", "w") as f:
                json.dump(state, f)
        except OSError as e:
            print(f"[SERVER] Impossible d'écrire round_state.json: {e}")

    def load_round_checkpoint(self):
        """Log checkpoint info on startup if a matching state file exists."""
        if not os.path.exists("round_state.json"):
            return
        try:
            with open("round_state.json") as f:
                state = json.load(f)
            if state.get("current_round") == self.current_round:
                print(f"[SERVER] Checkpoint trouvé pour le round {state['current_round']}, "
                      f"{state['num_received']} clients déjà reçus.")
        except Exception as e:
            print(f"[SERVER] Impossible de lire round_state.json: {e}")

    def GetModel(self, request, context):
        client_id = request.client_id

        # Simuler une perte de message sur GetModel
        if self.net_sim.should_drop("GetModel", client_id):
            api.global_state["metrics"]["messages_lost"] = self.net_sim.get_count()
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("[NETWORK SIM] Message perdu (simulé)")
            return fl_pb2.ModelWeights()

        t = self.clock.tick()
        return fl_pb2.ModelWeights(
            weights=serialize_weights(self.global_weights),
            round_num=self.current_round,
            lamport_time=t
        )

    def SendGradients(self, request, context):
        client_id = request.client_id

        # Simuler une perte de message sur SendGradients
        if self.net_sim.should_drop("SendGradients", client_id):
            api.global_state["metrics"]["messages_lost"] = self.net_sim.get_count()
            return fl_pb2.AggregationResult(success=False, next_round_num=self.current_round)

        weights = deserialize_weights(request.weights)

        # Mettre à jour l'horloge de Lamport
        t = self.clock.update(request.lamport_time)
        print(f"[LAMPORT] Gradients reçus de {client_id} à t={t}")
        api.global_state["metrics"]["lamport_time"] = t
        api.global_state["metrics"]["messages_lost"] = self.net_sim.get_count()

        with self.round_lock:
            self.received_updates.append(weights)
            self.received_sizes.append(request.dataset_size)
            self.received_losses.append(request.loss)
            self.received_accuracies.append(request.accuracy)
            print(f"[SERVER] Gradients reçus de {client_id} pour le round {self.current_round}.")
            self.save_round_checkpoint()

            if len(self.received_updates) >= self.min_fit_clients and not self.aggregating:
                self.aggregating = True
                print(f"[SERVER] Async aggregation triggered at round {self.current_round} with {len(self.received_updates)} clients")

                # Byzantine filtering before FedAvg
                filtered_weights, filtered_sizes = byzantine_filter(self.received_updates, self.received_sizes)
                excluded_count = len(self.received_updates) - len(filtered_weights)
                api.global_state["metrics"]["byzantine_excluded"] = excluded_count

                if len(filtered_weights) >= self.min_fit_clients:
                    self.global_weights = fedavg(filtered_weights, filtered_sizes)

                    # Mark clients that missed this round
                    active_clients = self.fault_detector.get_status()
                    for cid, info in active_clients.items():
                        if info.get("last_round") < self.current_round and info.get("status") != "INACTIVE":
                            self.fault_detector.record_missed_round(cid)

                    self.save_model(self.current_round)

                    avg_loss = sum(self.received_losses) / len(self.received_losses)
                    avg_accuracy = sum(self.received_accuracies) / len(self.received_accuracies)
                    duration_sec = time.time() - self._round_start_time

                    log_metrics(self.current_round, avg_loss, avg_accuracy, duration_sec, run_mode)
                    api.global_state["metrics"]["loss"] = avg_loss
                    api.global_state["metrics"]["accuracy"] = avg_accuracy
                    api.global_state["metrics"]["duration_sec"] = duration_sec
                    print(f"[SERVER] Round {self.current_round} — Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, Duration={duration_sec:.1f}s")

                    self.current_round += 1
                    api.global_state["current_round"] = self.current_round
                else:
                    print(f"[SERVER] AVERTISSEMENT: après filtrage byzantin, seulement {len(filtered_weights)} clients restants (min={self.min_fit_clients}). Agrégation ignorée.")

                # Reset state for next round
                self.received_updates = []
                self.received_sizes = []
                self.received_losses = []
                self.received_accuracies = []
                self._round_start_time = time.time()
                self.aggregating = False

        return fl_pb2.AggregationResult(success=True, next_round_num=self.current_round)

    def SendHeartbeat(self, request, context):
        self.fault_detector.update_heartbeat(request.client_id, request.status, self.current_round)
        return fl_pb2.HeartbeatAck(received=True)

    def start_training(self):
        print("[SERVER] Attente d'enregistrement des clients...")
        # Attente active jusqu'à ce que min_fit_clients soient enregistrés (timeout 10 min)
        deadline = time.time() + 600
        while len(self.clients) < self.min_fit_clients:
            if time.time() > deadline:
                print("[SERVER] Timeout: pas assez de clients enregistrés. Arrêt.")
                return
            print(f"[SERVER] {len(self.clients)}/{self.min_fit_clients} clients enregistrés, attente...")
            time.sleep(5)
        print(f"[SERVER] {len(self.clients)} clients prêts. Démarrage de l'entraînement asynchrone (mode={run_mode}).")

        # Initialise le timer et l'état API pour le round 1
        self._round_start_time = time.time()
        api.global_state["current_round"] = self.current_round

        # Lightweight monitor — aggregation fires inline inside SendGradients()
        while self.current_round <= self.num_rounds:
            time.sleep(1)
        print("[SERVER] Entraînement terminé !")

def serve():
    # Start Flask API in a separate thread
    flask_thread = threading.Thread(target=api.run_api, daemon=True)
    flask_thread.start()

    # Start gRPC Server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fl_server = FederatedServer()
    fl_pb2_grpc.add_FederatedServiceServicer_to_server(fl_server, server)
    
    port = os.environ.get("GRPC_PORT", 50051)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"[SERVER] Serveur gRPC démarré sur le port {port}.")
    
    fl_server.start_training()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
