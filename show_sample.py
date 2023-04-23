from src.game_client import GameClient
from utils import plot_sample
import random
from src.backends.backend import backend
import torch
import matplotlib.pyplot as plt

device = "cpu"
if torch.backends.mps.is_available():
    device = "ane"
if torch.cuda.is_available():
    device = "cuda:0"

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