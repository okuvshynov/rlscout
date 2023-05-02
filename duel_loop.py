import argparse
import numpy as np
import time
import logging
from threading import Thread

from src.backends.backend import backend
from src.rlslib import rlslib, EvalFn, LogFn, GameDoneFn
from src.game_client import GameClient
from src.utils import pick_device

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/duel_loop.log', level=logging.INFO)

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device', default=pick_device())
parser.add_argument('-m', '--model_server', default='tcp://localhost:8888')
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-g', '--games', type=int, default=128)
parser.add_argument('--rollouts', type=int, default=3000)
parser.add_argument('--raw_rollouts', type=int, default=3000)
parser.add_argument('--random_rollouts', type=int, default=20)
parser.add_argument('--iterations', type=int, default=20000)

args = parser.parse_args()

device = args.device
model_server = args.model_server
batch_size = args.batch_size
games_to_play = args.games
model_rollouts = args.rollouts
random_rollouts = args.random_rollouts
raw_rollouts = args.raw_rollouts
iterations = args.iterations

board_size = 6
margin = games_to_play // 16
explore_for_n_moves = 20
model_temp = 2.5
raw_temp = 1.5

games_done = 0
games_stats = {0: 0, -1: 0, 1:0}

client = GameClient(model_server)
boards_buffer = np.zeros(batch_size * 2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)
scores_buffer = np.ones(batch_size, dtype=np.float32)

log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

def start_batch_duel():
    global games_done, games_stats, start
    (model_to_eval_id, model_to_eval) = client.get_model_to_eval()
    if model_to_eval is not None:
        model_to_eval = backend(device, model_to_eval, batch_size, board_size)
    else:
        return False

    (best_model_id, best_model) = client.get_best_model()
    if best_model is not None:
        best_model = backend(device, best_model, batch_size, board_size)

    models_by_id = {
        best_model_id: best_model,
        model_to_eval_id: model_to_eval
    }

    def game_done_fn(score, game_id):
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
        logging.info(f'result = {score}|{winner}, done {local_gd} games. rate = {rate:.3f} games/s')

        # count done + enqueued
        return local_gd + batch_size <= games_to_play

    def eval_fn(model_id, add_noise_IGNORE):
        probs = models_by_id[model_id].get_probs(boards_buffer)
        np.copyto(probs_buffer, probs.reshape(
            (batch_size * board_size * board_size, )))

    def log_fn(game_id, player, skipped):
        pass

    logging.info(f'playing {model_to_eval_id} vs {best_model_id}')

    games_stats = {0: 0, -1: 0, 1:0}
    start = time.time()
    games_done = 0

    # new model first player
    rlslib.batch_mcts(
        batch_size,
        boards_buffer,
        probs_buffer,
        scores_buffer,
        log_boards_buffer,
        log_probs_buffer,
        EvalFn(eval_fn),
        LogFn(log_fn),
        GameDoneFn(game_done_fn),
        model_to_eval_id,
        best_model_id,
        explore_for_n_moves,
        model_rollouts,
        model_temp,
        model_rollouts if best_model is not None else raw_rollouts,
        model_temp if best_model is not None else raw_temp,
        random_rollouts,
        random_rollouts
    )
    logging.info(games_stats)

    local_stats = {
        'new': games_stats[0],
        'old': games_stats[1]
    }

    games_stats = {0: 0, -1: 0, 1:0}
    games_done = 0
    start = time.time()

    rlslib.batch_mcts(
        batch_size,
        boards_buffer,
        probs_buffer,
        scores_buffer,
        log_boards_buffer,
        log_probs_buffer,
        EvalFn(eval_fn),
        LogFn(log_fn),
        GameDoneFn(game_done_fn),
        best_model_id,
        model_to_eval_id,
        explore_for_n_moves,
        model_rollouts if best_model is not None else raw_rollouts,
        model_temp if best_model is not None else raw_temp,
        model_rollouts,
        model_temp,
        random_rollouts,
        random_rollouts
    )
    logging.info(games_stats)

    local_stats['new'] += games_stats[1]
    local_stats['old'] += games_stats[0]

    logging.info(local_stats)

    outcome = '+' if local_stats['new'] >= local_stats['old'] + margin else '-'
    client.record_eval(model_to_eval_id, outcome)

    return True

def duel_loop():
    played = 0
    while True:
        if not start_batch_duel():
            logging.info('no model to eval, sleeping')
            time.sleep(60)
        else:
            played += 1
            if played >= iterations:
                break

t = Thread(target=duel_loop, daemon=False)
t.start()
t.join()
