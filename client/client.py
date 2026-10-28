import grpc
import time
import os
import io
import random
import threading
import numpy as np
import torch
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
# pyrefly: ignore [missing-import]
import fl_pb2
# pyrefly: ignore [missing-import]
import fl_pb2_grpc

from model import SimpleCNN, train, test, train_fedprox, get_global_weights_as_tensors
from data_loader import load_partition
import hotel_api

# Horloge de Lamport locale (embarquée ici pour éviter les problèmes d'import)
class LamportClock:
    def __init__(self):
        self.time = 0
        self.lock = threading.Lock()
    def tick(self):
        with self.lock:
            self.time += 1
            return self.time
    def update(self, received_time):
        with self.lock:
            self.time = max(self.time, received_time) + 1
            return self.time
    def get(self):
        with self.lock:
            return self.time

def serialize_weights(weights):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *weights)
    buffer.seek(0)  # Rembobiner avant getvalue() pour s'assurer que tout est flush
    return buffer.getvalue()

def deserialize_weights(weights_bytes):
    buffer = io.BytesIO(weights_bytes)
    buffer.seek(0)  # S'assurer que le curseur est au début
    data = np.load(buffer, allow_pickle=False)
    # np.savez_compressed génère des clés "arr_0", "arr_1", ...
    return [data[key] for key in data.files]

class FlowerClient:
    def __init__(self, client_id, model, trainloader, testloader, num_examples):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, epochs=1):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        return self.get_parameters(), self.num_examples

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return loss, accuracy

    def fit_fedprox(self, parameters, global_tensors, epochs=1):
        """FedProx local training — keeps weights close to global model."""
        self.set_parameters(parameters)
        train_fedprox(self.model, self.trainloader, global_tensors, epochs=epochs, device=self.device)
        return self.get_parameters(), self.num_examples

def apply_differential_privacy(weights, clip_norm, epsilon):
    """Add calibrated Gaussian noise to weights for epsilon-DP (no external libs)."""
    sigma = clip_norm / epsilon
    noisy = []
    for w in weights:
        noise = np.random.normal(0, sigma, w.shape).astype(w.dtype)
        noisy.append(w + noise)
    return noisy

def heartbeat_loop(stub, client_id, get_status):
    attempt = 0
    while True:
        try:
            status = get_status()
            stub.SendHeartbeat(fl_pb2.Heartbeat(
                client_id=str(client_id),
                timestamp=int(time.time()),
                status=status
            ))
            attempt = 0
            time.sleep(5)
        except grpc.RpcError:
            print("[CLIENT] Échec du heartbeat. Tentative de reconnexion...")
            delay = min(60, 2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
            attempt += 1

def run():
    client_id = int(os.environ.get("CLIENT_ID", 0))
    partition_type = os.environ.get("PARTITION_TYPE", "iid")
    seed = int(os.environ.get("SEED", 42))
    server_address = os.environ.get("SERVER_ADDRESS", "fl-server:50051")
    hotel_name = os.environ.get("HOTEL", f"Hotel-{client_id}")
    dp_epsilon = float(os.environ.get("DP_EPSILON", "1.0"))
    dp_clip_norm = float(os.environ.get("DP_CLIP_NORM", "1.0"))
    USE_FEDPROX = os.environ.get("USE_FEDPROX", "true").lower() == "true"
    
    print(f"[CLIENT {client_id}] Démarrage pour {hotel_name} (Partition: {partition_type})...")

    # Load data
    trainloader, testloader, num_examples = load_partition(client_id, partition_type, seed)
    model = SimpleCNN()
    
    # Initialize FlowerClient logic
    client = FlowerClient(client_id, model, trainloader, testloader, num_examples)

    # Status state
    current_status = "READY"
    def get_status(): return current_status

    channel = grpc.insecure_channel(server_address)
    stub = fl_pb2_grpc.FederatedServiceStub(channel)

    # Register
    attempt = 0
    while True:
        try:
            resp = stub.Register(fl_pb2.ClientInfo(client_id=str(client_id)))
            if resp.success:
                print(f"[CLIENT {client_id}] Enregistré avec succès.")
                attempt = 0  # reset on success
                break
        except grpc.RpcError:
            delay = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"[CLIENT {client_id}] En attente du serveur... (retry {attempt}, attente {delay:.1f}s)")
            time.sleep(delay)
            attempt += 1

    # Initialise hotel API shared state
    hotel_api.shared_state["hotel_name"] = hotel_name
    hotel_api.shared_state["client_id"]  = client_id
    hotel_api.shared_state["status"]     = "READY"

    # Start hotel API Flask server in a daemon thread
    flask_thread = threading.Thread(target=hotel_api.run_hotel_api, daemon=True)
    flask_thread.start()
    print(f"[CLIENT {client_id}] Hotel API démarrée sur le port 8000.")

    # Start Heartbeat thread
    hb_thread = threading.Thread(target=heartbeat_loop, args=(stub, client_id, get_status), daemon=True)
    hb_thread.start()

    current_round = 1
    max_rounds = int(os.environ.get("NUM_ROUNDS", 20))
    clock = LamportClock()  # Horloge de Lamport locale
    
    attempt = 0
    while current_round <= max_rounds:
        try:
            attempt = 0
            current_status = "READY"
            print(f"[CLIENT {client_id}] En attente du modèle global pour le round {current_round}...")
            
            # Polling to check if server has advanced to current_round
            while True:
                model_resp = stub.GetModel(fl_pb2.RoundRequest(client_id=str(client_id), requested_round=current_round))
                # Mettre à jour l'horloge de Lamport depuis la réponse du serveur
                if model_resp.lamport_time > 0:
                    t = clock.update(model_resp.lamport_time)
                    print(f"[LAMPORT] GetModel reçu, horloge locale à t={t}")
                if model_resp.round_num == current_round:
                    break
                # If server is at a higher round (we missed one), we catch up
                if model_resp.round_num > current_round:
                    print(f"[CLIENT {client_id}] Le serveur est au round {model_resp.round_num}, on rattrape.")
                    current_round = model_resp.round_num
                    break
                time.sleep(2)
            
            # Start Training
            current_status = "TRAINING"
            global_weights = deserialize_weights(model_resp.weights)
            
            print(f"[CLIENT {client_id}] Entraînement local (Round {current_round})...")
            print(f"[CLIENT {client_id}] Mode: {'FedProx' if USE_FEDPROX else 'FedAvg'}")

            if USE_FEDPROX:
                global_tensors = get_global_weights_as_tensors(global_weights)
                updated_weights, dataset_size = client.fit_fedprox(global_weights, global_tensors, epochs=1)
            else:
                updated_weights, dataset_size = client.fit(global_weights, epochs=1)

            # Apply Differential Privacy before sending weights
            # DP disabled for now — uncomment the lines below to re-enable
            # sigma = dp_clip_norm / dp_epsilon
            # updated_weights = apply_differential_privacy(updated_weights, dp_clip_norm, dp_epsilon)
            # print(f"[CLIENT {client_id}] DP applied: epsilon={dp_epsilon}, sigma={sigma:.4f}")
            
            # Evaluation on local test set
            loss, accuracy = client.evaluate(updated_weights)
            print(f"[CLIENT {client_id}] Eval locale: Loss={loss:.4f}, Acc={accuracy:.4f}")

            # Update hotel API shared state so /predict uses the latest model
            hotel_api.shared_state["model"]         = client.model
            hotel_api.shared_state["current_round"] = current_round
            hotel_api.shared_state["accuracy"]      = accuracy
            hotel_api.shared_state["loss"]           = loss
            hotel_api.shared_state["status"]         = current_status
            
            # Envoyer les gradients avec l'horloge de Lamport
            t = clock.tick()
            print(f"[LAMPORT] SendGradients à t={t}")
            stub.SendGradients(fl_pb2.GradientUpdate(
                weights=serialize_weights(updated_weights),
                dataset_size=dataset_size,
                client_id=str(client_id),
                loss=loss,
                accuracy=accuracy,
                lamport_time=t
            ))
            print(f"[CLIENT {client_id}] Gradients envoyés pour le round {current_round}.")
            
            current_status = "DONE"
            current_round += 1
            
        except grpc.RpcError as e:
            delay = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"[CLIENT {client_id}] Erreur gRPC, reconnexion dans {delay:.1f}s... {e}")
            time.sleep(delay)
            attempt += 1
            continue
        attempt = 0  # reset backoff counter on a successful round iteration

    # Training finished — update status and keep the process alive so the
    # Hotel API Flask thread (daemon) continues serving /predict and /dashboard.
    hotel_api.shared_state["status"] = "DONE"
    print(f"[CLIENT {client_id}] Entraînement FL terminé. Hotel API reste disponible sur le port 8000.")
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    run()
