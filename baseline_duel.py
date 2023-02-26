import argparse
import numpy as np
import time
import torch
import sys

from backends.backend import backend
from batch_mcts import batch_mcts_lib, EvalFn, LogFn, BoolFn
from game_client import GameClient

device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')
parser.add_argument('-m', '--model_id')

args = parser.parse_args()

if args.device is not None:
    device = args.device

if args.model_id is not None:
    model_id = int(args.model_id)

board_size = 6
batch_size = 32
games_done = 0
games_to_play = 128
games_stats = {0: 0, -1: 0, 1:0}
model_rollouts = 5000
model_temp = 4.0

raw_rollouts = 5000
raw_temp = 1.5

client = GameClient()
boards_buffer = np.zeros(batch_size * 2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)

log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

def start_batch_duel(model_id):
    global games_done, games_stats, start
    model_to_eval = client.get_model(model_id)
    if model_to_eval is not None:
        model_to_eval = backend(device, model_to_eval, batch_size, board_size)
    else:
        return False

    models_by_id = {
        model_id: model_to_eval
    }

    def game_done_fn(winner):
        global games_done

        games_done += 1
        games_stats[winner] += 1
        local_gd = games_done

        rate = 1.0 * local_gd / (time.time() - start)
        print(f'result = {winner}, done {local_gd} games. rate = {rate:.3f} games/s')

        # count done + enqueued
        return local_gd + batch_size <= games_to_play

    def eval_fn(model_id):
        probs = models_by_id[model_id].get_probs(boards_buffer)
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))

    def log_fn(model_id):
        pass

    print(f'playing {model_id} vs baseline {raw_rollouts} rollouts')

    games_stats = {0: 0, -1: 0, 1:0}
    start = time.time()
    games_done = 0

    # new model first player
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
        0, # model_id
        0, #explore_for_n_moves,
        model_rollouts,
        model_temp,
        raw_rollouts,
        raw_temp
    )
    print(games_stats)

    local_stats = {
        'new': games_stats[0],
        'old': games_stats[1]
    }

    games_stats = {0: 0, -1: 0, 1:0}
    games_done = 0
    start = time.time()

    batch_mcts_lib.batch_mcts(
        batch_size,
        boards_buffer,
        probs_buffer,
        log_boards_buffer,
        log_probs_buffer,
        EvalFn(eval_fn),
        LogFn(log_fn),
        BoolFn(game_done_fn),
        0, # model id
        model_id,
        0, #explore_for_n_moves,
        raw_rollouts,
        raw_temp,
        model_rollouts,
        model_temp,
    )
    print(games_stats)

    local_stats['new'] += games_stats[1]
    local_stats['old'] += games_stats[0]

    print(local_stats)

    return True

if __name__ == '__main__':
    start_batch_duel(model_id)