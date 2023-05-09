import argparse
import numpy as np
from utils.utils import pick_device
from utils.backends.backend import backend
from utils.game_client import GameClient

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('--model_server', default='tcp://localhost:8888')
parser.add_argument('--model_id', type=int, required=True)
parser.add_argument('-d', '--device', default=pick_device())
args = parser.parse_args()

device = args.device
model_server = args.model_server

client = GameClient(model_server)

batch_size = 1
board_size = 6

boards_buffer = np.zeros(batch_size * 2 * board_size *
                        board_size, dtype=np.int32)

model = backend(device, client.get_model(args.model_id), batch_size=batch_size, board_size=board_size)

print(model.get_probs(boards_buffer))
