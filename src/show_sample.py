import matplotlib.pyplot as plt
import random
import torch

from src.backends.backend import backend
from src.game_client import GameClient
from src.utils import pick_device

def plot_sample(where, board, probs):
    m = board.shape[1]
    n = board.shape[2]
    for x in range(m):
        for y in range(n):
            stone = -1
            if board[0, y, x] > 0:
                stone = 0
            if board[1, y, x] > 0:
                stone = 1

            ch = '0' if stone == 0 else 'X'
            if stone >= 0:
                where.text(x, y, ch, weight="bold", color="red",
                    fontsize='large', va='center', ha='center')
    where.imshow(probs.view(m, n).cpu().numpy(), cmap='Blues')

device = pick_device()

client = GameClient()

samples = client.get_batch(1000)

model_a = client.get_model(69)
model_b = client.get_model(127)

model_a = backend(device, model_a, 1, 6)
model_b = backend(device, model_b, 1, 6)

random.shuffle(samples)

def format_sample(sample):
    _, v, b, p, _, _= sample
    return b.view(2, 6, 6), p, v

for s in samples:
    f, axarr = plt.subplots(3,1) 
    b, p, v = format_sample(s)
    pa, va = model_a.get_probs(b)
    pb, vb = model_b.get_probs(b)

    print(f'values: {v} {va} {vb}')

    plot_sample(axarr[0], b, p)
    plot_sample(axarr[1], b, torch.tensor(pa))
    plot_sample(axarr[2], b, torch.tensor(pb))
    plt.show()
