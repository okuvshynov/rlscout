import argparse
import random
import sys
import time
import torch
import torch.optim as optim

from action_value_model import ActionValueModel
from game_client import GameClient

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda:0"

parser = argparse.ArgumentParser("rlscout training")
parser.add_argument('-d', '--device')

args = parser.parse_args()

if args.device is not None:
    device = args.device

minibatch_size = 512
epochs = 20
minibatch_per_epoch = 100
checkpoints = 1000
max_samples = 50000
min_samples = 16000

random.seed(1991)

gen = torch.Generator()
gen.manual_seed(1991)

# let's get samples:

client = GameClient()

(last_model_id, action_model) = client.get_last_model()

print(f'loading last snapshot from DB: id={last_model_id}')
print(f'training on device {device}')

if action_model is None:
    action_model = ActionValueModel()

action_model = action_model.to(device)

optimizer = optim.SGD(action_model.parameters(), lr=0.001, momentum=0.9)

# expects tensor of shape [?, N, N], returns list of 8 tensors
def symm(t):
    res = [torch.rot90(t, w, [1, 2]) for w in range(4)]
    t = torch.flip(t, [1])
    res += [torch.rot90(t, w, [1, 2]) for w in range(4)]
    return res

def train_minibatch(boards, probs):
    # pick minibatch
    ix = torch.randint(0, boards.shape[0], (minibatch_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]

    optimizer.zero_grad()
    actions_probs = action_model(X)

    pb = y.view(y.shape[0], -1)
    loss = -torch.mean(torch.sum(pb * actions_probs, dim=1))
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_sample(boards, probs):
    sample_size = 2 ** 14
    # pick sample
    ix = torch.randint(0, boards.shape[0], (sample_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    with torch.no_grad():
        action_probs = action_model(X)
        probs = y.view(y.shape[0], -1)
        loss = -torch.mean(torch.sum(probs * action_probs, dim=1))
    return loss.item()

for checkpoint in range(checkpoints):
    samples = client.get_batch(max_samples)
    if len(samples) < min_samples:
        print('Not enough samples in the DB. Waiting for 3 minutes.')
        time.sleep(3 * 60)
        continue

    samples_symm = []
    for b, p in samples:
        if torch.isnan(b).any() or torch.isnan(p).any():
            continue
        if torch.isinf(b).any() or torch.isinf(p).any():
            continue
        samples_symm.extend(list(zip(symm(b), symm(p))))

    print(f'training on {len(samples_symm)} recent samples')

    random.shuffle(samples_symm)

    boards, probs = zip(*samples_symm)

    idx = int(0.8 * len(boards))

    boards_train = torch.stack(boards[:idx]).float()
    boards_val = torch.stack(boards[idx:]).float()

    probs_train = torch.stack(probs[:idx]).float()
    probs_val = torch.stack(probs[idx:]).float()

    boards_train_dev = boards_train.to(device)
    probs_train_dev = probs_train.to(device)
    boards_val_dev = boards_val.to(device)
    probs_val_dev = probs_val.to(device)

    epoch_train_losses = [evaluate_sample(boards_train_dev, probs_train_dev)]
    epoch_validation_losses = [evaluate_sample(boards_val_dev, probs_val_dev)]
    print(f'training loss: {epoch_train_losses[-1]:.3f}')
    print(f'validation loss: {epoch_validation_losses[-1]:.3f}')

    for e in range(epochs):
        start = time.time()
        for i in range(minibatch_per_epoch):
            train_minibatch(boards_train_dev, probs_train_dev)
            if i % 10 == 9:
                sys.stdout.write('.')
                sys.stdout.flush()
        #if e == reduce_rate_at:
        #    print(f'reducing learning rate')
        #    optimizer.param_groups[0]['lr'] = 0.0005
        epoch_train_losses.append(evaluate_sample(boards_train_dev, probs_train_dev))
        epoch_validation_losses.append(evaluate_sample(boards_val_dev, probs_val_dev))
        dur = time.time() - start
        print(f' | {dur:.1f} seconds | epoch {e}: training loss: {epoch_train_losses[-1]:.3f}, validation loss: {epoch_validation_losses[-1]:.3f}')
    
    print('saving model snapshot')
    print(client.save_model_snapshot(action_model))
