import argparse
import random
import sys
import time
import torch
import torch.optim as optim
from collections import deque

from action_value_model import ActionValueModel
from src.game_client import GameClient

random.seed(1991)

gen = torch.Generator()
gen.manual_seed(1991)

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda:0"

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')
parser.add_argument('-s', '--data_server')
parser.add_argument('-m', '--model_server')

args = parser.parse_args()

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

(last_model_id, action_model) = model_client.get_last_model()
print(f'loading last snapshot from DB: id={last_model_id}')
print(f'training on device {device}')

sample_id = 0
read_batch_size = 2 ** 12

current_data = []

epoch_samples_max = 2 ** 20
epoch_samples_min = 2 ** 15

train_set_rate = 0.8

minibatch_per_epoch = 3000
minibatch_size = 512

# circular buffer for 'most recent training samples'
current_samples = deque([], maxlen=epoch_samples_max)

#if action_model is None:
action_model = ActionValueModel()
    
action_model = action_model.to(device)

optimizer = optim.SGD(action_model.parameters(), lr=0.001, momentum=0.9)

score_loss_fn = torch.nn.MSELoss()

def sign(v):
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0

# expects tensor of shape [?, N, N], returns list of 8 tensors
def symm(t):
    res = [torch.rot90(t, w, [1, 2]) for w in range(4)]
    t = torch.flip(t, [1])
    res += [torch.rot90(t, w, [1, 2]) for w in range(4)]
    return res

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
    loss = action_loss + score_loss

    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_sample(boards, probs, scores):
    sample_size = 2 ** 14
    # pick sample
    ix = torch.randint(0, boards.shape[0], (sample_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    z = scores[ix]

    with torch.no_grad():
        action_probs, score = action_model(X)
        probs = y.view(y.shape[0], -1)
        action_loss = -torch.mean(torch.sum(probs * action_probs, dim=1))
        score_loss = score_loss_fn(z, score.view(-1))
        print(f'action loss: {action_loss}, score loss: {score_loss}')
        loss = action_loss + score_loss
        
    return loss.item()

e = 0

while True:
    # read samples 
    while True:
        batch = data_client.get_batch(read_batch_size, sample_id)
        if len(batch) == 0:
            # TODO: wait? do what?
            break

        max_id = max(s_id for s_id, _, _, _, _, _ in batch)
        sample_id = max(sample_id, max_id)
        current_samples.extend(batch)
        print(f'sample size: {len(current_samples)}')

        if len(batch) < read_batch_size:
            break

    if len(current_samples) < epoch_samples_min:
        print(f'not enough samples to continue training: {len(current_samples)}')
        time.sleep(60)
        continue

    nans = 0

    samples = []

    for s_id, score, b, p, player, skipped in current_samples:
        if torch.isnan(b).any() or torch.isnan(p).any():
            nans += 1
            continue
        if torch.isinf(b).any() or torch.isinf(p).any():
            nans += 1
            continue

        # game was not finished and we dodn't record the score
        if score is None:
            nans += 1
            continue

        value = sign(score)

        if player == 1:
            value = - value

        # boards are ordered from POV of current player, but score is
        # from player 0 POV.
        samples.extend(list(zip(symm(b), symm(p), [value] * 8)))
    random.shuffle(samples)

    boards, probs, scores = zip(*samples)
    scores = list(scores)

    idx = int(train_set_rate * len(boards))

    boards_train = torch.stack(boards[:idx]).float().to(device)
    boards_val = torch.stack(boards[idx:]).float().to(device)

    print(boards_train.shape)

    probs_train = torch.stack(probs[:idx]).float().to(device)
    probs_val = torch.stack(probs[idx:]).float().to(device)

    scores_train = torch.tensor(scores[:idx]).float().to(device)
    scores_val = torch.tensor(scores[idx:]).float().to(device)

    print(scores_train.shape)

    start = time.time()
    for i in range(minibatch_per_epoch):
        train_minibatch(boards_train, probs_train, scores_train)
        if i % 10 == 9:
            sys.stdout.write('.')
            sys.stdout.flush()
        if i % 100 == 99:
            dur = time.time() - start
            train_loss = evaluate_sample(boards_train, probs_train, scores_train)
            val_loss = evaluate_sample(boards_val, probs_val, scores_val)
            print(f' | {dur:.1f} seconds | epoch {e}: training loss: {train_loss:.3f}, validation loss: {val_loss:.3f}')

    print('saving model snapshot')
    model_client.save_model_snapshot(action_model)
