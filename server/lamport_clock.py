import threading


class LamportClock:
    """
    Horloge logique de Lamport pour ordonner les événements dans un système distribué.
    Règle : à la réception d'un message de timestamp T, l'horloge locale devient max(local, T) + 1.
    """

    def __init__(self):
        self.time = 0
        self.lock = threading.Lock()

    def tick(self):
        """Événement local : incrémente l'horloge et retourne la nouvelle valeur."""
        with self.lock:
            self.time += 1
            return self.time

    def update(self, received_time):
        """
        Réception d'un message estampillé received_time.
        Applique la règle de Lamport : local = max(local, received) + 1.
        """
        with self.lock:
            self.time = max(self.time, received_time) + 1
            return self.time

    def get(self):
        """Retourne la valeur courante sans l'incrémenter."""
        with self.lock:
            return self.time
