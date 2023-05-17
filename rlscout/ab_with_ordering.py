import argparse
import numpy as np
import logging
from threading import Thread
import random
import torch

from utils.backends.backend import backend
from rlslib.rlscout_native import RLScoutNative, EvalFn
from utils.game_client import GameClient
from utils.utils import pick_device, parse_ids, random_seed, symm
from utils.show_sample import compare_probs
from prometheus_client import Counter, Gauge
from prometheus_client import start_http_server

## TODO:
# - make it a loop as well
# - write results to database. Results include both absolute time and number of visits per level.

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/ab_ordering.log', level=logging.INFO)

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device', default=pick_device())
parser.add_argument('-s', '--model_server', default='tcp://localhost:8888')
parser.add_argument('-m', '--model_ids')
parser.add_argument('--seed', type=int, default=random_seed())

args = parser.parse_args()

device = args.device
model_server = args.model_server

# prometheus
eval_counter = Counter('model_inference', 'how many eval calls were there')
visits_gauge = Gauge('alpha_beta_visits_counter', 'how many new nodes were visited', labelnames=['model_id'])
start_http_server(9006)

board_size = 6

client = GameClient(model_server)
boards_buffer = np.zeros(2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

if args.model_ids is None:
    models = [client.get_best_model()]
else:
    models = [(model_id, client.get_model(model_id)) for model_id in parse_ids(args.model_ids)]

random.seed(args.seed)
rng = np.random.default_rng(seed=args.seed)
rls_native = RLScoutNative(seed=args.seed)

for (model_id, model) in models:
    logging.info(f'starting guided alpha-beta search with model_id={model_id}')
    model1 = backend(device, model, batch_size=1, board_size=board_size)
    #model8 = backend(device, model, batch_size=8, board_size=board_size)

    def eval_fn(model_id_IGNORE, add_noise_IGNORE):
        eval_counter.inc()
        # get all symmetries
        #symm_boards = symm(torch.from_numpy(boards_buffer).reshape(2, board_size, board_size))
        #symm_boards = torch.stack(symm_boards).numpy()
        probs1 = model1.get_probs(boards_buffer)
        #probs8 = model8.get_probs(symm_boards.reshape((8 * 2 * board_size * board_size, )))
        #symm_probs = symm(torch.from_numpy(probs1).reshape(1, board_size, board_size))
        #print(probs8[0].numpy())
        #print(symm_probs[0])
        #for i in range(8):
        #    pa = torch.from_numpy(probs8[i])
        #    pb = symm_probs[i]
        #    compare_probs(pa, pb)
        
        np.copyto(probs_buffer, probs1.reshape(
            (board_size * board_size, )))
        
    def start_ab():
        total_visits = rls_native.lib.run_ab(boards_buffer, probs_buffer, EvalFn(eval_fn), -5, -3)
        logging.info(f'observed total visits = {total_visits} for model_id={model_id}')
        visits_gauge.labels(f'{model_id}').set(total_visits)

    t = Thread(target=start_ab, daemon=False)
    t.start()
    t.join()
