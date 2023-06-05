import argparse
import random
import time
import torch
import torch.optim as optim
import logging
from prometheus_client import Counter, Gauge
from prometheus_client import start_http_server

from model.action_value_model import ActionValueModel
from utils.game_client import GameClient
from utils.utils import pick_train_device, random_seed
from utils.data_reader import DataReader, IncrementalDataReader

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/training_loop.log', level=logging.INFO)

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
parser.add_argument('--wait_for_evaluation', type=int, default=5)
parser.add_argument('--evaluation_sample_size', type=int, default=2**14)
parser.add_argument('--snapshots', type=int, default=100000)
parser.add_argument('--value_weight', type=float, default=0.0)

args = parser.parse_args()

random.seed(random_seed())
gen = torch.manual_seed(random_seed())

device = args.device
data_server = args.data_server
model_server = args.model_server

# prometheus
snapshot_counter = Counter('snapshot_saved', 'how many snapshots were saved')
samples_gauge = Gauge('samples_loaded', 'how many samples were loaded')
loss_gauge = Gauge('loss_observed', 'observed loss', labelnames=['dataset'])
start_http_server(9004)

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
value_weight = args.value_weight

if action_model is None:
    action_model = ActionValueModel(n=6, m=6, channels=128, nblocks=6)
action_model = action_model.to(device)

optimizer = optim.SGD(action_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
score_loss_fn = torch.nn.MSELoss()

def train_minibatch(boards, probs, scores):
    # pick minibatch
    ix = torch.randint(0, boards.shape[0], (minibatch_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    z = scores[ix]

    optimizer.zero_grad()
    actions_probs, score = action_model(X)

    pb = y.view(y.shape[0], -1)
    action_loss = -torch.mean(torch.sum(pb * actions_probs, dim=1))
    score_loss = score_loss_fn(z, score.view(-1))
    loss = action_loss + value_weight * score_loss

    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_sample(boards, probs, scores):
    # pick sample
    ix = torch.randint(0, boards.shape[0], (evaluation_sample_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    z = scores[ix]

    with torch.no_grad():
        action_probs, score = action_model(X)
        probs = y.view(y.shape[0], -1)
        action_loss = -torch.mean(torch.sum(probs * action_probs, dim=1))
        score_loss = score_loss_fn(z, score.view(-1))
        logging.info(f'action loss: {action_loss}, score loss: {score_loss}')
        #loss = action_loss + score_loss
        
    return action_loss.item(), score_loss.item()

# reader = DataReader(data_client, train_set_rate, epoch_samples_max)
reader = IncrementalDataReader(data_client, train_set_rate)
wait_s = 60

snapshot = 0 
while True:
    samples = reader.read_samples()
    if samples is None:
        samples_gauge.set(0)
        logging.info(f'no samples, waiting for {wait_s} seconds')
        time.sleep(wait_s)
        continue

    (boards_train, probs_train, scores_train, boards_val, probs_val, scores_val) = samples
    samples_gauge.set(boards_train.shape[0])

    boards_train = boards_train.to(device)
    boards_val = boards_val.to(device)
    probs_train = probs_train.to(device)
    probs_val = probs_val.to(device)
    scores_train = scores_train.to(device)
    scores_val = scores_val.to(device)

    models_to_eval = model_client.count_models_to_eval()
    if models_to_eval > wait_for_evaluation:
        logging.info(f'{models_to_eval} models are not evaluated yet, waiting for {wait_s} seconds')
        time.sleep(wait_s)
        continue

    if boards_train.shape[0] < epoch_samples_min:
        logging.info(f'{boards_train.shape} samples only, waiting for {wait_s} seconds')
        time.sleep(wait_s)
        continue

    logging.info(f'starting epoch on {boards_train.shape} samples and {boards_val.shape} validation set')

    start = time.time()
    for i in range(minibatch_per_epoch):
        train_minibatch(boards_train, probs_train, scores_train)
        if i % 100 == 99:
            dur = time.time() - start
            actions_loss, value_loss = evaluate_sample(boards_train, probs_train, scores_train)
            loss_gauge.labels('train_actions_loss').set(actions_loss)
            loss_gauge.labels('train_value_loss').set(value_loss)
            train_loss = actions_loss + value_loss

            actions_loss, value_loss = evaluate_sample(boards_val, probs_val, scores_val)
            loss_gauge.labels('validation_actions_loss').set(actions_loss)
            loss_gauge.labels('validation_value_loss').set(value_loss)
            val_loss = actions_loss + value_loss
            logging.info(f'{dur:.1f} seconds | minibatches {i + 1} | training loss: {train_loss:.3f}, validation loss: {val_loss:.3f}')

    logging.info(f'saving model snapshot {snapshot}')
    snapshot += 1
    model_client.save_model_snapshot(action_model)
    snapshot_counter.inc()
    if snapshot >= snapshots:
        break
