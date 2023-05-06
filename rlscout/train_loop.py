import argparse
import random
import time
import torch
import torch.optim as optim
import logging

from model.action_value_model import ActionValueModel
from utils.game_client import GameClient
from utils.utils import pick_train_device
from utils.data_reader import DataReader

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/training_loop.log', level=logging.INFO)

random.seed(1991)

gen = torch.Generator()
gen.manual_seed(1991)

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device', default=pick_train_device())
parser.add_argument('-s', '--data_server', default='tcp://localhost:8889')
parser.add_argument('-m', '--model_server', default='tcp://localhost:8888')
parser.add_argument('-f', '--from_model_id')
parser.add_argument('--read_batch_size', type=int, default=2**13)
parser.add_argument('--epoch_samples_max', type=int, default=2**20)
parser.add_argument('--epoch_samples_min', type=int, default=2**18)
parser.add_argument('--dataset_split', type=float, default=0.8)
parser.add_argument('--minibatch_per_epoch', type=int, default=5000)
parser.add_argument('--minibatch_size', type=int, default=512)
parser.add_argument('--wait_for_evaluation', type=int, default=2)
parser.add_argument('--evaluation_sample_size', type=int, default=2**14)
parser.add_argument('--snapshots', type=int, default=100000)

args = parser.parse_args()

device = args.device
data_server = args.data_server
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

read_batch_size = args.read_batch_size
epoch_samples_max = args.epoch_samples_max
epoch_samples_min = args.epoch_samples_min
train_set_rate = args.dataset_split
minibatch_per_epoch = args.minibatch_per_epoch
minibatch_size = args.minibatch_size
wait_for_evaluation = args.wait_for_evaluation
read_batch_size = args.read_batch_size
evaluation_sample_size = args.evaluation_sample_size
snapshots = args.snapshots

if action_model is None:
    action_model = ActionValueModel()
action_model = action_model.to(device)

optimizer = optim.SGD(action_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
# score_loss_fn = torch.nn.MSELoss()

def train_minibatch(boards, probs):
    # pick minibatch
    ix = torch.randint(0, boards.shape[0], (minibatch_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    #z = scores[ix]

    optimizer.zero_grad()
    actions_probs = action_model(X)

    pb = y.view(y.shape[0], -1)
    action_loss = -torch.mean(torch.sum(pb * actions_probs, dim=1))
    score_loss = 0 # score_loss_fn(z, score.view(-1))
    loss = action_loss + score_loss

    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_sample(boards, probs):
    # pick sample
    ix = torch.randint(0, boards.shape[0], (evaluation_sample_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    #z = scores[ix]

    with torch.no_grad():
        action_probs = action_model(X)
        probs = y.view(y.shape[0], -1)
        action_loss = -torch.mean(torch.sum(probs * action_probs, dim=1))
        score_loss = 0 # score_loss_fn(z, score.view(-1))
        #logging.info(f'action loss: {action_loss}, score loss: {score_loss}')
        loss = action_loss + score_loss
        
    return loss.item()

reader = DataReader(data_client, train_set_rate, epoch_samples_max)
wait_s = 60

snapshot = 0 
while True:

    samples = reader.read_samples()
    if samples is None:
        logging.info(f'no samples, waiting for {wait_s} seconds')
        time.sleep(wait_s)
        continue

    (boards_train, probs_train, boards_val, probs_val) = samples

    boards_train = boards_train.to(device)
    boards_val = boards_val.to(device)
    probs_train = probs_train.to(device)
    probs_val = probs_val.to(device)

    models_to_eval = model_client.count_models_to_eval()
    if models_to_eval > wait_for_evaluation:
        logging.info(f'{models_to_eval} models are not evaluated yet, waiting for {wait_s} seconds')
        time.sleep(wait_s)
        continue

    if boards_train.shape[0] < epoch_samples_min:
        logging.info(f'{boards_train.shape} samples only, waiting for {wait_s} seconds')
        time.sleep(wait_s)
        continue

    start = time.time()
    for i in range(minibatch_per_epoch):
        train_minibatch(boards_train, probs_train)
        if i % 100 == 99:
            dur = time.time() - start
            train_loss = evaluate_sample(boards_train, probs_train)
            val_loss = evaluate_sample(boards_val, probs_val)
            logging.info(f'{dur:.1f} seconds | minibatches {i + 1} | training loss: {train_loss:.3f}, validation loss: {val_loss:.3f}')

    logging.info(f'saving model snapshot {snapshot}')
    snapshot += 1
    model_client.save_model_snapshot(action_model)
    if snapshot >= snapshots:
        break
