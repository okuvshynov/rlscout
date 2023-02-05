import ctypes
import numpy as np
import numpy.ctypeslib as ctl
import os
from players import CoreMLGameModel
from game_client import GameClient
from threading import Thread, Lock
import time
import torch

from numpy.ctypeslib import ndpointer

board_size = 8
batch_size = 16
nthreads = 8

batch_mcts = ctl.load_library("libmcts.so", os.path.join(
    os.path.dirname(__file__), "mnklib", "_build"))

VoidFn = ctypes.CFUNCTYPE(ctypes.c_void_p)
batch_mcts.batch_mcts.argtypes = [
    ctypes.c_int, 
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    VoidFn,
    VoidFn,
    VoidFn
]
batch_mcts.batch_mcts.restype = None

client = GameClient()  # default localhost:8888

(_, best_model) = client.get_best_model()

core_ml_model = CoreMLGameModel(best_model, batch_size=batch_size)

## callbacks from native land

games_done = 0
games_done_lock = Lock()
start = time.time()

def game_done_fn():
    global games_done, games_done_lock

    with games_done_lock:
        games_done += 1
        local_gd = games_done
    rate = 1.0 * local_gd / (time.time() - start)
    print(f'done {local_gd} games. rate = {rate:.3f} games/s')


def start_batch_mcts():
    boards_buffer = np.zeros(batch_size * 2 * board_size *
                            board_size, dtype=np.int32)
    probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)

    log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
    log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

    client = GameClient()

    def eval_fn():
        probs = core_ml_model.get_probs(boards_buffer)
        #print(probs)
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))

    # TODO: not thread safe?
    def log_fn():
        board = torch.from_numpy(log_boards_buffer).float()
            
        # log probs
        prob = torch.from_numpy(log_probs_buffer)
        prob = prob / prob.sum()
            
        # TODO: add model id here
        client.append_sample(board, prob.view(1, board_size, board_size), 0)

    batch_mcts.batch_mcts(
        batch_size,
        boards_buffer,
        probs_buffer,
        log_boards_buffer,
        log_probs_buffer,
        VoidFn(eval_fn),
        VoidFn(log_fn),
        VoidFn(game_done_fn)
    )


threads = [Thread(target=start_batch_mcts, daemon=False)
           for _ in range(nthreads)]

for t in threads:
    t.start()

for t in threads:
    t.join()
