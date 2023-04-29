import argparse
import random
import time
import torch
import torch.optim as optim
from collections import deque
import logging

from src.action_value_model import ActionValueModel
from src.game_client import GameClient
from src.utils import pick_train_device
from src.data_reader import DataReader

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/training_loop.log', encoding='utf-8', level=logging.INFO)

random.seed(1991)

gen = torch.Generator()
gen.manual_seed(1991)

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')
parser.add_argument('-s', '--data_server')
parser.add_argument('-m', '--model_server')
parser.add_argument('-f', '--from_model_id')

args = parser.parse_args()

device = pick_train_device()
if args.device is not None:
    device = args.device

data_server = 'tcp://localhost:8889'
if args.data_server is not None:
    data_server = args.data_server

model_server = 'tcp://localhost:8888'
if args.model_server is not None:
    model_server = args.model_server

model_client = GameClient(model_server)
data_client = GameClient(data_server)

if args.from_model_id is None:
    (last_model_id, action_model) = model_client.get_last_model()
else:
    last_model_id = int(args.from_model_id)
    action_model = model_client.get_model(last_model_id)

logging.info(f'loading snapshot from DB: id={last_model_id}')
logging.info(f'training on device {device}')

sample_id = 0
read_batch_size = 2 ** 13

current_data = []

epoch_samples_max = 2 ** 20
epoch_samples_min = 2 ** 18

train_set_rate = 0.8

minibatch_per_epoch = 5000
minibatch_size = 512

wait_for_evaluation = 2

# circular buffer for 'most recent training samples'
current_samples = deque([], maxlen=epoch_samples_max)

if action_model is None:
    action_model = ActionValueModel()
    
action_model = action_model.to(device)

optimizer = optim.SGD(action_model.parameters(), lr=0.001, momentum=0.9)

score_loss_fn = torch.nn.MSELoss()

def train_minibatch(boards, probs):
    # pick minibatch
    ix = torch.randint(0, boards.shape[0], (minibatch_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    #z = scores[ix]

    optimizer.zero_grad()
    actions_probs, _ = action_model(X)

    pb = y.view(y.shape[0], -1)
    action_loss = -torch.mean(torch.sum(pb * actions_probs, dim=1))
    score_loss = 0 # score_loss_fn(z, score.view(-1))
    loss = action_loss + score_loss

    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_sample(boards, probs):
    sample_size = 2 ** 14
    # pick sample
    ix = torch.randint(0, boards.shape[0], (sample_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    #z = scores[ix]

    with torch.no_grad():
        action_probs, _ = action_model(X)
        probs = y.view(y.shape[0], -1)
        action_loss = -torch.mean(torch.sum(probs * action_probs, dim=1))
        score_loss = 0 # score_loss_fn(z, score.view(-1))
        #logging.info(f'action loss: {action_loss}, score loss: {score_loss}')
        loss = action_loss + score_loss
        
    return loss.item()

e = 0

reader = DataReader(data_client, read_batch_size, device)

while True:
    reader.read_samples()

    models_to_eval = model_client.count_models_to_eval()
    if models_to_eval > wait_for_evaluation:
        logging.info(f'{models_to_eval} models are not evaluated yet, waiting')
        time.sleep(60)
        continue

    if reader.boards_train.shape[0] < epoch_samples_min:
        logging.info(f'{reader.boards_train.shape} samples only, waiting')
        time.sleep(60)
        continue

    start = time.time()
    for i in range(minibatch_per_epoch):
        train_minibatch(reader.boards_train, reader.probs_train)
        if i % 100 == 99:
            dur = time.time() - start
            train_loss = evaluate_sample(reader.boards_train, reader.probs_train)
            val_loss = evaluate_sample(reader.boards_val, reader.probs_val)
            logging.info(f'{dur:.1f} seconds | minibatches {e}:{i + 1} | training loss: {train_loss:.3f}, validation loss: {val_loss:.3f}')

    logging.info('saving model snapshot')
    model_client.save_model_snapshot(action_model)
