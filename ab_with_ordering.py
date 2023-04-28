import argparse
import numpy as np
import time
import torch
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/ab_ordering.log', encoding='utf-8', level=logging.INFO)

from src.backends.backend import backend
from src.batch_mcts import batch_mcts_lib, EvalFn
from src.game_client import GameClient

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

client = GameClient(model_server)
boards_buffer = np.zeros(2 * board_size *
                        board_size, dtype=np.int32)
probs_buffer = np.ones(board_size * board_size, dtype=np.float32)

(best_model_id, best_model) = client.get_best_model()
best_model = backend(device, best_model, batch_size=1, board_size=board_size)

def eval_fn(model_id_IGNORE, add_noise_IGNORE):
    probs, _ = best_model.get_probs(boards_buffer)
    np.copyto(probs_buffer, probs.reshape(
        (board_size * board_size, )))

print(batch_mcts_lib.run_ab(boards_buffer, probs_buffer, EvalFn(eval_fn), -5, -3))
