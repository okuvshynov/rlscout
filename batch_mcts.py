import ctypes
import numpy as np
import numpy.ctypeslib as ctl
import os
from players import CoreMLGameModel
from game_client import GameClient
from threading import Thread, Lock
import time

from numpy.ctypeslib import ndpointer

board_size = 8
batch_size = 32
nthreads = 2

batch_mcts = ctl.load_library("libmcts.so", os.path.join(
                os.path.dirname(__file__), "mnklib", "_build"))

EvalFunction = ctypes.CFUNCTYPE(ctypes.c_void_p)
batch_mcts.batch_mcts.argtypes = [ctypes.c_int, ndpointer(
    ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), EvalFunction, EvalFunction, EvalFunction]
batch_mcts.batch_mcts.restype = None


client = GameClient() # default localhost:8888

(_, best_model) = client.get_best_model()

core_ml_model = CoreMLGameModel(best_model, batch_size=batch_size)
boards_buffer = np.zeros(batch_size * 2 * board_size * board_size, dtype=np.int32)
probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)

def eval_fn():
    probs = core_ml_model.get_probs(boards_buffer)
    #print(probs)
    np.copyto(probs_buffer, probs.reshape((batch_size * board_size * board_size, )))
    #print('called eval')
    pass

def log_fn():
    pass


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
    batch_mcts.batch_mcts(batch_size, boards_buffer, probs_buffer, EvalFunction(eval_fn), EvalFunction(log_fn), EvalFunction(game_done_fn))

threads = [Thread(target=start_batch_mcts, daemon=False) for _ in range(nthreads)]

for t in threads:
    t.start()

for t in threads:
    t.join()