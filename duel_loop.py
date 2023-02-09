import numpy as np
from players import CoreMLGameModel
from game_client import GameClient
import time
from batch_mcts import batch_mcts_lib, EvalFn, LogFn, BoolFn

board_size = 8
batch_size = 8
games_done = 0
games_to_play = 64
margin = games_to_play // 16
games_stats = {0: 0, -1: 0, 1:0}
explore_for_n_moves = 0
model_rollouts = 1000
model_temp = 4.0

raw_rollouts = 500000
raw_temp = 1.5

client = GameClient()
boards_buffer = np.zeros(batch_size * 2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)

log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

def start_batch_duel():
    global games_done, games_stats, start
    (model_to_eval_id, model_to_eval) = client.get_model_to_eval()
    if model_to_eval is not None:
        model_to_eval = CoreMLGameModel(model_to_eval, batch_size=batch_size)
    else:
        return False

    (best_model_id, best_model) = client.get_best_model()
    if best_model is not None:
        best_model = CoreMLGameModel(best_model, batch_size=batch_size)

    models_by_id = {
        best_model_id: best_model,
        model_to_eval_id: model_to_eval
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

    print(f'playing {model_to_eval_id} vs {best_model_id}')

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
        model_to_eval_id,
        best_model_id,
        explore_for_n_moves,
        model_rollouts,
        model_temp,
        model_rollouts if best_model is not None else raw_rollouts,
        model_temp if best_model is not None else raw_temp
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
        best_model_id,
        model_to_eval_id,
        explore_for_n_moves,
        model_rollouts if best_model is not None else raw_rollouts,
        model_temp if best_model is not None else raw_temp,
        model_rollouts,
        model_temp,
    )
    print(games_stats)

    local_stats['new'] += games_stats[1]
    local_stats['old'] += games_stats[0]

    print(local_stats)

    outcome = '+' if local_stats['new'] >= local_stats['old'] + margin else '-'
    client.record_eval(model_to_eval_id, outcome)

    return True

while True:
    if not start_batch_duel():
        print('no model to eval, sleeping')
        time.sleep(60)