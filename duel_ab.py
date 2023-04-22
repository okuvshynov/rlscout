from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
import numpy as np
import time
import torch
from collections import defaultdict

from src.backends.backend import backend
from src.batch_mcts import batch_duel_lib, EvalFn, GameDoneFn
from src.game_client import GameClient

# can be 'cpu', 'cuda:x', 'mps', 'ane'
device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

board_size = 6
batch_size = 128

games_done = 0
games_done_lock = Lock()
start = time.time()
games_to_play = 256
games_stats = defaultdict(lambda : 0)
explore_for_n_moves = 1
model_rollouts = 1000
model_temp = 2.5


client = GameClient()
boards_buffer = np.zeros(batch_size * 2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)
scores_buffer = np.ones(batch_size, dtype=np.float32)

def start_batch_duel():
    global games_done, games_stats, start

    model_id, model = client.get_best_model()
    if model is None:
        return False

    model = backend(device, model, batch_size, board_size)

    def game_done_fn(score, game_id_IGNORE):
        global games_done

        winner = -1
        if score > 0:
            winner = 0
        if score < 0:
            winner = 1

        games_done += 1
        games_stats[winner] += 1
        local_gd = games_done

        rate = 1.0 * local_gd / (time.time() - start)
        print(f'result = {score}|{winner}, done {local_gd} games. rate = {rate:.3f} games/s')
        print(games_stats)
        # count done + enqueued
        return local_gd + batch_size <= games_to_play

    def eval_fn(model_id_IGNORE, add_noise_IGNORE):
        probs, scores = model.get_probs(boards_buffer)
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))
        np.copyto(scores_buffer, scores.reshape(
            (batch_size, )))

    batch_duel_lib.ab_duel(
        batch_size,
        boards_buffer,
        probs_buffer,
        scores_buffer,
        EvalFn(eval_fn),
        GameDoneFn(game_done_fn),
        model_id,
        explore_for_n_moves,
        model_rollouts,
        model_temp,
        -5,
        5,
        13
    )
    print(games_stats)

if not start_batch_duel():
    print('no model to eval, sleeping')