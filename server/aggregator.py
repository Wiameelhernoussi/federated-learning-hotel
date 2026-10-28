import numpy as np
import io
import pandas as pd
import os

def fedavg(weights_results, sizes):
    """
    weights_results: liste de listes de numpy arrays (poids de chaque client)
    sizes: liste des tailles de dataset correspondantes
    Retourne la liste des numpy arrays agrégés
    """
    total_size = sum(sizes)
    if total_size == 0:
        return None

    # Initialisation de la moyenne pondérée avec le premier client
    weighted_weights = [
        weights_results[0][layer] * (sizes[0] / total_size)
        for layer in range(len(weights_results[0]))
    ]

    # Ajout des pondérations des autres clients
    for i in range(1, len(weights_results)):
        for layer in range(len(weights_results[i])):
            weighted_weights[layer] += weights_results[i][layer] * (sizes[i] / total_size)

    return weighted_weights

def byzantine_filter(weights_results, sizes, threshold_factor=10):
    """
    Exclude clients whose L2 norm of concatenated weights is abnormally large.
    Returns filtered (weights_results, sizes) tuple.
    """
    norms = []
    for w_list in weights_results:
        flat = np.concatenate([w.flatten() for w in w_list])
        norms.append(np.linalg.norm(flat))

    median_norm = np.median(norms)
    threshold = threshold_factor * median_norm

    filtered_weights = []
    filtered_sizes = []
    for i, (w_list, size) in enumerate(zip(weights_results, sizes)):
        if norms[i] > threshold:
            print(f"[BYZANTINE FILTER] Client index {i} excluded, norm={norms[i]:.2f} > threshold={threshold:.2f}")
        else:
            filtered_weights.append(w_list)
            filtered_sizes.append(size)

    return filtered_weights, filtered_sizes

def serialize_weights(weights):
    """Convertit une liste de numpy arrays en bytes."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *weights)
    buffer.seek(0)  # Rembobiner pour s'assurer que getvalue() lit tout
    return buffer.getvalue()

def deserialize_weights(weights_bytes):
    """Convertit des bytes en liste de numpy arrays."""
    buffer = io.BytesIO(weights_bytes)
    buffer.seek(0)  # Sécurité : curseur au début avant np.load
    data = np.load(buffer, allow_pickle=False)
    # np.savez_compressed génère des clés "arr_0", "arr_1", ...
    return [data[key] for key in data.files]

def log_metrics(round_num, loss, accuracy, duration_sec, mode="iid", filename="metrics.csv"):
    df = pd.DataFrame([{
        "round": round_num,
        "loss": loss,
        "accuracy": accuracy,
        "duration": duration_sec,
        "mode": mode
    }])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
