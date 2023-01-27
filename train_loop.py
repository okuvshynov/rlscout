from action_value_model import ActionValueModel

import random
import sys
import torch
import torch.optim as optim
from local_db import LocalDB
from io import BytesIO
import time

device = "mps"
minibatch_size = 512
epochs = 10
minibatch_per_epoch = 100
checkpoints = 1000
min_samples = 20000

random.seed(1991)

gen = torch.Generator()
gen.manual_seed(1991)

# let's get samples:

db = LocalDB('./_out/8x8/test2.db')

action_model = ActionValueModel().to(device)

optimizer = optim.Adam(action_model.parameters(), weight_decay=0.001, lr=0.005)


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
    sample_size = 2 ** 12
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
    data = db.get_batch(min_samples)
    if len(data) < min_samples:
        print('Not enough samples in the DB. Waiting for a minute.')
        time.sleep(60)
        continue

    unpack = lambda buf: torch.load(BytesIO(buf))

    print(unpack(data[0][0]))

    boards, probs = zip(*[(unpack(b), unpack(p)) for (b, p) in data])

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
    
    model_buffer = BytesIO()
    torch.save(action_model, model_buffer)
    print('saving model snapshot')
    db.save_snapshot(model_buffer.getvalue())