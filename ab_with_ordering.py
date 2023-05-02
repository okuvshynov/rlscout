import argparse
import numpy as np
import logging
from threading import Thread

from src.backends.backend import backend
from src.rlslib import rlslib, EvalFn
from src.game_client import GameClient
from src.utils import pick_device

## TODO:
# - make it a loop as well
# - write results to database. Results include both absolute time and number of visits per level.
# - change logging to PyLog

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/ab_ordering.log', level=logging.INFO)

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device', default=pick_device())
parser.add_argument('-s', '--model_server', default='tcp://localhost:8888')
parser.add_argument('-m', '--model_id')

args = parser.parse_args()

device = args.device
model_server = args.model_server

board_size = 6

client = GameClient(model_server)
boards_buffer = np.zeros(2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

if args.model_id is None:
    (best_model_id, model) = client.get_best_model()
else:
    model = client.get_model(args.model_id)

model = backend(device, model, batch_size=1, board_size=board_size)

def eval_fn(model_id_IGNORE, add_noise_IGNORE):
    probs = model.get_probs(boards_buffer)
    np.copyto(probs_buffer, probs.reshape(
        (board_size * board_size, )))
    
def start_ab():
    rlslib.run_ab(boards_buffer, probs_buffer, EvalFn(eval_fn), -5, -3)

t = Thread(target=start_ab, daemon=False)
t.start()
t.join()

