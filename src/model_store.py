from threading import Lock
import time
from src.backends.backend import backend
import logging

class ModelStore:
    def __init__(self, client, device, board_size, batch_size=1):
        self.lock = Lock()
        self.model_id = 0
        self.model = None
        self.game_client = client
        self.batch_size = batch_size
        self.board_size = board_size
        self.last_refresh = 0.0
        self.device = device
        self.maybe_refresh_model()

    # loads new model if different from current
    def maybe_refresh_model(self):
        with self.lock:
            if self.last_refresh + 2.0 > time.time():
                # no refreshing too often
                return
            out = self.game_client.get_best_model()
            self.last_refresh = time.time()
            (model_id, torch_model) = out
            if model_id == self.model_id:
                return 
            model = backend(self.device, torch_model, self.batch_size, self.board_size)
            (self.model_id, self.model) = (model_id, model)
            logging.info(f'new best model: {self.model_id}')

    def get_best_model(self):
        with self.lock:
            return (self.model_id, self.model)