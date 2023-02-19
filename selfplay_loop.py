import numpy as np
from game_client import GameClient
from threading import Thread, Lock
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from batch_mcts import batch_mcts_lib, EvalFn, LogFn, BoolFn
from backend import backend
import argparse
from batch_evaluator import BatchEvaluator
from threading import Condition

# can be 'cpu', 'cuda:x', 'mps', 'ane'
device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')

args = parser.parse_args()

if args.device is not None:
    device = args.device

board_size = 8
minibatch_size = 64
batch_size = 256
nthreads = 8
games_done = 0
games_done_lock = Lock()
start = time.time()
games_to_play = 100000
games_stats = {0: 0, -1: 0, 1:0}
explore_for_n_moves = 8
model_rollouts = 1000
model_temp = 4.0

raw_rollouts = 500000
raw_temp = 1.5

evaluator = BatchEvaluator(None, minibatch_size, batch_size, board_size)

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
            if self.last_refresh + 1.0 > time.time():
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
    boards_buffer = np.zeros(minibatch_size * 2 * board_size *
                            board_size, dtype=np.int32)
    probs_buffer = np.ones(minibatch_size * board_size * board_size, dtype=np.float32)

    log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
    log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

    client = GameClient()

    model_id, model = models.get_best_model()

    models_by_id = {
        model_id: model
    }

    evaluator.backend = model
    cv = Condition()

    def game_done_fn(winner):
        global games_done, games_done_lock

        with games_done_lock:
            games_done += 1
            games_stats[winner] += 1
            local_gd = games_done

        models.maybe_refresh_model()
        model_id, model = models.get_best_model()

        nonlocal models_by_id
        models_by_id = {
            model_id: model
        }
        rate = 1.0 * local_gd / (time.time() - start)
        print(f'result = {winner}, done {local_gd} games. rate = {rate:.3f} games/s')

        # count done + enqueued
        return local_gd + minibatch_size * nthreads<= games_to_play

    def eval_fn(model_id):
        evaluator.enqueue_and_wait(boards_buffer, probs_buffer, cv)

    def log_fn(model_id):
        board = torch.from_numpy(log_boards_buffer).float()
            
        prob = torch.from_numpy(log_probs_buffer)
        prob = prob / prob.sum()
            
        client.append_sample(board.view(2, board_size, board_size), prob.view(1, board_size, board_size), model_id)

    batch_mcts_lib.batch_mcts(
        minibatch_size,
        boards_buffer,
        probs_buffer,
        log_boards_buffer,
        log_probs_buffer,
        EvalFn(eval_fn),
        LogFn(log_fn),
        BoolFn(game_done_fn),
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
