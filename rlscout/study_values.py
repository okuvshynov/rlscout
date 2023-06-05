import torch
import torch.optim as optim
import numpy as np
import random
import time

from model.action_value_model import ActionValueModel

device = 'mps'
gen = torch.manual_seed(286749)
minibatch_size = 128
evaluation_sample_size = 2**14
minibatch_per_epoch = 5000

scores_unique = torch.load('db/scores_train_unique.pt')
boards_unique = torch.load('db/boards_train_unique.pt')

action_model = ActionValueModel(n=6, m=6, channels=64, nblocks=2, hidden_fc=32)
action_model = action_model.to(device)

optimizer = optim.SGD(action_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
score_loss_fn = torch.nn.MSELoss()

def train_minibatch(boards, scores):
    # pick minibatch
    ix = torch.randint(0, boards.shape[0], (minibatch_size, ), generator=gen)
    X = boards[ix]
    z = scores[ix]

    optimizer.zero_grad()
    _, score = action_model(X)

    loss = score_loss_fn(z, score.view(-1))
    
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_sample(boards, scores):
    # pick sample
    ix = torch.randint(0, boards.shape[0], (evaluation_sample_size, ), generator=gen)
    X = boards[ix]
    z = scores[ix]

    with torch.no_grad():
        _, score = action_model(X)
        score_loss = score_loss_fn(z, score.view(-1))
        
    return score_loss.item()

def evaluate_all(boards, scores):
    with torch.no_grad():
        _, score = action_model(boards)
        score_loss = score_loss_fn(scores, score.view(-1))
        
    return score_loss.item()

print(scores_unique.shape, boards_unique.shape)

samples = list(zip(list(boards_unique), list(scores_unique)))
random.shuffle(samples)

split = int(boards_unique.shape[0] * 0.9)
boards_train, scores_train = zip(*samples[:split])
boards_val, scores_val = zip(*samples[split:])

boards_train = torch.stack(boards_train).to(device)
boards_val = torch.stack(boards_val).to(device)
scores_train = torch.stack(scores_train).to(device)
scores_val = torch.stack(scores_val).to(device)

print(boards_train.shape)

while True:
    start = time.time()
    for i in range(minibatch_per_epoch):
        train_minibatch(boards_train, scores_train)
        if i % 100 == 99:
            dur = time.time() - start
            train_loss = evaluate_sample(boards_train, scores_train)
            val_loss = evaluate_all(boards_val, scores_val)
            print(f'{dur:.1f} seconds | minibatches {i + 1} | training loss: {train_loss:.3f}, validation loss: {val_loss:.3f}')
