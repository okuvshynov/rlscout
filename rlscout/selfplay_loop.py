from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
import argparse
import numpy as np
import time
import torch
from collections import defaultdict
import logging
import random
from prometheus_client import Counter
from prometheus_client import start_http_server


#from rlslib.rlslib import rlslib, EvalFn, LogFn, GameDoneFn
from rlslib.rlscout_native import RLScoutNative, EvalFn, LogFn, GameDoneFn, ModelIDFn
from utils.game_client import GameClient
from utils.model_store import ModelStore
from utils.utils import pick_device, random_seed

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/selfplay_loop.log', level=logging.INFO)

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device', default=pick_device())
parser.add_argument('-t', '--nthreads', type=int, default=1)
parser.add_argument('-s', '--data_server', default='tcp://localhost:8889')
parser.add_argument('-m', '--model_server', default='tcp://localhost:8888')
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-g', '--games', type=int, default=1024)
parser.add_argument('--rollouts', type=int, default=3000)
parser.add_argument('--random_rollouts', type=int, default=20)
parser.add_argument('--seed', type=int, default=random_seed())
args = parser.parse_args()

device = args.device
data_server = args.data_server
model_server = args.model_server
nthreads = args.nthreads
batch_size = args.batch_size
games_to_play = args.games

# prometheus
eval_counter = Counter('model_inference', 'how many eval calls were there')
games_counter = Counter('games_done', 'how many games were finished')
start_http_server(9002)

board_size = 6
explore_for_n_moves = 20
model_rollouts = args.rollouts
model_temp = 2.5
random_rollouts = args.random_rollouts
dirichlet_noise = 0.3

games_done = 0
games_done_lock = Lock()
start = time.time()
games_stats = defaultdict(lambda : 0)

random.seed(args.seed)
rng = np.random.default_rng(seed=args.seed)
rls_native = RLScoutNative(seed=args.seed)

def add_dirichlet_noise(probs, eps):
    alpha = np.ones_like(probs) * 0.3
    noise = rng.dirichlet(alpha) * batch_size
    res = (1 - eps) * probs + eps * noise
    return res

logging.info(f'starting self-play on device {device}')
executor = ThreadPoolExecutor(max_workers=1)
log_executor = ThreadPoolExecutor(max_workers=1)

logging.info('setting up model store')
models = ModelStore(GameClient(model_server), device=device, batch_size=batch_size, board_size=board_size)

def start_batch_mcts():
    boards_buffer = np.zeros(batch_size * 2 * board_size *
                            board_size, dtype=np.int32)
    probs_buffer = np.ones(batch_size * board_size * board_size, dtype=np.float32)
    scores_buffer = np.ones(batch_size, dtype=np.float32)

    log_boards_buffer = np.zeros(2 * board_size * board_size, dtype=np.int32)
    log_probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

    client = GameClient(data_server)

    model_id, model = models.get_best_model()

    def game_done_fn(score, game_id):
        global games_done, games_done_lock
        nonlocal model_id
        games_counter.inc()

        with games_done_lock:
            games_done += 1
            games_stats[score] += 1
            local_gd = games_done

        nonlocal model
        models.maybe_refresh_model()
        model_id, model = models.get_best_model()
        rate = 1.0 * local_gd / (time.time() - start)
        logging.info(f'result = {score}, done {local_gd} games. rate = {rate:.3f} games/s')

        def log_game_done_impl(score, game_id):
            client.game_done(game_id, score)

        log_executor.submit(log_game_done_impl, score, game_id)

        # count done + enqueued
        return local_gd + batch_size * nthreads <= games_to_play

    def eval_fn(model_id_IGNORE, add_noise):
        eval_counter.inc()
        def eval_model():
            if model is not None:
                return model.get_probs(boards_buffer)
        fut = executor.submit(eval_model)
        probs = fut.result()
        probs = probs.reshape((batch_size * board_size * board_size, ))
        if add_noise:
            probs = add_dirichlet_noise(probs, dirichlet_noise)
        np.copyto(probs_buffer, probs)

    def log_fn(game_id, player, skipped):
        ## logging will be done in separate thread so we clone 
        board = torch.from_numpy(log_boards_buffer).clone()
        prob = torch.from_numpy(log_probs_buffer).clone()
        def log_impl(board, prob, game_id, player, skipped, key):
            board = board.float()
            prob = prob / prob.sum()
            client.append_sample(board.view(2, board_size, board_size), prob.view(1, board_size, board_size), game_id, player, skipped, key)

        log_executor.submit(log_impl, board, prob, game_id, player, skipped, random.randint(-2**63, 2**63 - 1))

    rls_native.lib.batch_mcts(
        batch_size,
        boards_buffer,
        probs_buffer,
        scores_buffer,
        log_boards_buffer,
        log_probs_buffer,
        EvalFn(eval_fn),
        LogFn(log_fn),
        GameDoneFn(game_done_fn),
        ModelIDFn(lambda: model_id),
        ModelIDFn(lambda: model_id),
        explore_for_n_moves,
        model_rollouts,
        model_temp,
        model_rollouts,
        model_temp,
        random_rollouts,
        random_rollouts
    )


if __name__ == '__main__':
    threads = [Thread(target=start_batch_mcts, daemon=False)
            for _ in range(nthreads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    logging.info(games_stats)