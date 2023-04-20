from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
import numpy as np
import time
import torch
from collections import defaultdict

from src.backends.backend import backend
from src.batch_mcts import batch_mcts_duel_lib, EvalFn, LogFn, GameDoneFn
from src.game_client import GameClient

# can be 'cpu', 'cuda:x', 'mps', 'ane'
device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

nthreads = 1

board_size = 6
batch_size = 256

games_done = 0
games_done_lock = Lock()
start = time.time()
games_to_play = 100000
games_stats = defaultdict(lambda : 0)
explore_for_n_moves = 1
model_rollouts = 1000
model_temp = 2.5

executor = ThreadPoolExecutor(max_workers=1)
log_executor = ThreadPoolExecutor(max_workers=1)

class ModelStore:
    def __init__(self, batch_size):
        self.lock = Lock()
        self.model_id = 0
        self.model = None
        self.game_client = GameClient()
        self.batch_size = batch_size
        self.last_refresh = 0.0
        self.maybe_refresh_model()

    # loads new model if different from current
    def maybe_refresh_model(self):
        global device
        with self.lock:
            if self.last_refresh + 2.0 > time.time():
                # no refreshing too often
                return
            out = self.game_client.get_best_model()
            self.last_refresh = time.time()
            (model_id, torch_model) = out
            if model_id == self.model_id:
                return 
            model = backend(device, torch_model, self.batch_size, board_size)
            (self.model_id, self.model) = (model_id, model)
            print(f'new best model: {self.model_id}')

    def get_best_model(self):
        with self.lock:
            return (self.model_id, self.model)

models = ModelStore(batch_size=batch_size)

def start_batch_mcts():
    boards_buffer = np.zeros(batch_size * 2 * board_size *
                            board_size, dtype=np.int32)
    probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)
    scores_buffer = np.ones(batch_size, dtype=np.float32)

    log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
    log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

    model_id, model = models.get_best_model()

    def log_fn(game_id, player, skipped):
        pass

    def game_done_fn(score, game_id):
        return False

    def eval_fn(model_id_IGNORE):
        # in self-play we ignore model id and just use the latest one

        def eval_model():
            if model is not None:
                return model.get_probs(boards_buffer)
        fut = executor.submit(eval_model)
        probs, scores = fut.result()
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))
        np.copyto(scores_buffer, scores.reshape(
            (batch_size, )))


    batch_mcts_duel_lib.batch_duel(
        batch_size,
        boards_buffer,
        probs_buffer,
        scores_buffer,
        log_boards_buffer,
        log_probs_buffer,
        EvalFn(eval_fn),
        LogFn(log_fn),
        GameDoneFn(game_done_fn),
        model_id,
        model_id,
        explore_for_n_moves,
        model_rollouts,
        model_temp,
        model_rollouts,
        model_temp
    )

threads = [Thread(target=start_batch_mcts, daemon=False)
           for _ in range(nthreads)]

for t in threads:
    t.start()

for t in threads:
    t.join()

print(games_stats)
