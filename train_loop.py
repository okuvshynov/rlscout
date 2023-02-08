from action_value_model import ActionValueModel
import random
import sys
import torch
import torch.optim as optim
import time
from game_client import GameClient
from utils import symm

device = "mps"
minibatch_size = 512
epochs = 10
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

if action_model is None:
    action_model = ActionValueModel()

action_model = action_model.to(device)

optimizer = optim.Adam(action_model.parameters(), weight_decay=0.001, lr=0.005)

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
        samples_symm.extend(list(zip(symm(b), symm(p))))

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
    print(f'training loss: {epoch_train_losses[-1]}')
    print(f'validation loss: {epoch_validation_losses[-1]}')

    for e in range(epochs):
        sys.stdout.write(f'{e}:0 ')
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
        print(f' | epoch {e}: training loss: {epoch_train_losses[-1]}, validation loss: {epoch_validation_losses[-1]}')
    
    print('saving model snapshot')
    print(client.save_model_snapshot(action_model))