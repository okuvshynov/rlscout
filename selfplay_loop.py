from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
import argparse
import numpy as np
import time
import torch

from backends.backend import backend
from batch_mcts import batch_mcts_lib, EvalFn, LogFn, BoolFn
from game_client import GameClient

# can be 'cpu', 'cuda:x', 'mps', 'ane'
device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')
parser.add_argument('-t', '--nthreads')

args = parser.parse_args()

if args.device is not None:
    device = args.device

nthreads = 1
if args.nthreads is not None:
    nthreads = int(args.nthreads)

board_size = 6
batch_size = 16

games_done = 0
games_done_lock = Lock()
start = time.time()
games_to_play = 100000
games_stats = {0: 0, -1: 0, 1:0}
explore_for_n_moves = 10
model_rollouts = 500000
model_temp = 1.5

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

    log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
    log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

    client = GameClient()

    model_id, model = models.get_best_model()

    def game_done_fn(winner):
        global games_done, games_done_lock

        with games_done_lock:
            games_done += 1
            games_stats[winner] += 1
            local_gd = games_done

        nonlocal model
        models.maybe_refresh_model()
        model_id, model = models.get_best_model()
        rate = 1.0 * local_gd / (time.time() - start)
        print(f'result = {winner}, done {local_gd} games. rate = {rate:.3f} games/s')

        # count done + enqueued
        return local_gd + batch_size * nthreads <= games_to_play

    def eval_fn(model_id_IGNORE):
        # in self-play we ignore model id and just use the latest one

        def eval_model():
            if model is not None:
                return model.get_probs(boards_buffer)
        fut = executor.submit(eval_model)
        probs = fut.result()
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))

    def log_fn(model_id_IGNORE):
        ## logging will be done in separate thread so we clone 
        board = torch.from_numpy(log_boards_buffer).clone()
        prob = torch.from_numpy(log_probs_buffer).clone()
        def log_impl(board, prob):
            board = board.float()
            prob = prob / prob.sum()
            client.append_sample(board.view(2, board_size, board_size), prob.view(1, board_size, board_size), model_id)

        log_executor.submit(log_impl, board, prob)

    batch_mcts_lib.batch_mcts(
        batch_size,
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
