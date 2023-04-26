import numpy as np
import time
import torch
import argparse
from collections import defaultdict

from src.backends.backend import backend
from src.batch_mcts import batch_mcts_lib, EvalFn, GameDoneFn
from src.game_client import GameClient

import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/duel_with_ab.log', encoding='utf-8', level=logging.INFO)

# can be 'cpu', 'cuda:x', 'mps', 'ane'
device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')
parser.add_argument('-s', '--model_server')

args = parser.parse_args()

if args.device is not None:
    device = args.device

model_server = 'tcp://localhost:8888'
if args.model_server is not None:
    model_server = args.model_server

board_size = 6
batch_size = 16

start = time.time()
games_to_play = 32

## MCTS config
explore_for_n_moves = 1
model_rollouts = 1000
model_temp = 2.5
random_rollouts = 18

## alpha-beta config
alpha = -5
beta = -3
full_search_after_move = 20

client = GameClient(model_server)
boards_buffer = np.zeros(batch_size * 2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)
scores_buffer = np.ones(batch_size, dtype=np.float32)

def start_batch_duel():
    global start
    games_done = 0
    games_stats = defaultdict(lambda : 0)

    model_id, model = client.get_best_model()
    if model is None:
        return False

    model = backend(device, model, batch_size, board_size)

    def game_done_fn(score, game_id_IGNORE):
        nonlocal games_done

        winner = -1
        if score > 0:
            winner = 0
        if score < 0:
            winner = 1

        games_done += 1
        games_stats[winner] += 1
        local_gd = games_done

        rate = 1.0 * local_gd / (time.time() - start)
        logging.info(f'result = {score}|{winner}, done {local_gd} games. rate = {rate:.3f} games/s')
        # count done + enqueued
        return local_gd + batch_size <= games_to_play

    def eval_fn(model_id_IGNORE, add_noise_IGNORE):
        probs, scores = model.get_probs(boards_buffer)
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))
        np.copyto(scores_buffer, scores.reshape(
            (batch_size, )))

    batch_mcts_lib.ab_duel(
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
        alpha,
        beta,
        full_search_after_move,
        False,
        random_rollouts
    )

    local_stats = {
        'new': games_stats[0],
        'old': games_stats[1]
    }
    games_stats = defaultdict(lambda : 0)
    games_done = 0

    batch_mcts_lib.ab_duel(
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
        alpha,
        beta,
        full_search_after_move,
        True,
        random_rollouts
    )

    local_stats['new'] += games_stats[1]
    local_stats['old'] += games_stats[0]

    logging.info(local_stats)

for iter in range(32):
    logging.info(f'starting iter {iter}')
    start_batch_duel()
    curr = time.time()
    dur = curr - start
    logging.info(f'done in {dur:.2f}')
    start = curr

if not start_batch_duel():
    logging.info('no model to eval, sleeping')