#from action_model import ActionNN
from action_model import ActionNN
import matplotlib.pyplot as plt
import random
import sys
import torch
import torch.optim as optim

boards = torch.load("./_out/boards_500000r_1000g.pt").float()
probs = torch.load("./_out/probs_500000r_1000g.pt")

device = "mps"
minibatch_size = 512
epochs = 200
minibatch_per_epoch = 100
data_subset = 100

random.seed(1991)

gen = torch.Generator()
gen.manual_seed(1991)

def plot_sample(board, probs):
    m = board.shape[1]
    n = board.shape[2]
    plt.figure(figsize=(3, 3))
    for x in range(m):
      for y in range(n):
        stone = -1
        if board[0, y, x] > 0:
          stone = 0
        if board[1, y, x] > 0:
          stone = 1

        ch = '0' if stone == 0 else 'X'
        if stone >= 0:
          plt.text(x, y, ch, weight="bold", color="red", fontsize='xx-large', va='center', ha='center')
    plt.imshow(probs.view(m, n).cpu().numpy(), cmap='Blues')
    plt.show()

# expects tensor of shape [?, N, N], returns list of 8 tensors
def symm(t):
  res = [torch.rot90(t, w, [1, 2]) for w in range(4)]
  t = torch.flip(t, [1])
  res += [torch.rot90(t, w, [1, 2]) for w in range(4)]
  return res

samples = []

for s in zip(boards.tolist(), probs.tolist()):
  p = torch.Tensor(s[1])
  p = p / p.sum()
  samples.extend(list(zip(symm(torch.Tensor(s[0])), symm(p.view(1, 7, 7)))))

random.shuffle(samples)

boards_symm, probs_sym = zip(*samples)

idx = int(0.8 * len(boards_symm))

boards_train = torch.stack(boards_symm[:idx])
boards_val = torch.stack(boards_symm[idx:])

probs_train = torch.stack(probs_sym[:idx])
probs_val = torch.stack(probs_sym[idx:])

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
    sample_size = 2 ** 15
    # pick sample
    ix = torch.randint(0, boards.shape[0], (sample_size, ), generator=gen)
    X = boards[ix]
    y = probs[ix]
    with torch.no_grad():
        action_probs = action_model(X)
        probs = y.view(y.shape[0], -1)
        loss = -torch.mean(torch.sum(probs * action_probs, dim=1))
    return loss.item()

action_model = ActionNN().to(device)

optimizer = optim.Adam(action_model.parameters(), weight_decay=0.001, lr=0.005)

boards_train_gpu = boards_train.to(device)
probs_train_gpu = probs_train.to(device)
boards_val_gpu = boards_val.to(device)
probs_val_gpu = probs_val.to(device)

epoch_train_losses = [evaluate_sample(boards_train_gpu, probs_train_gpu)]
epoch_validation_losses = [evaluate_sample(boards_val_gpu, probs_val_gpu)]
print(f'training loss: {epoch_train_losses[-1]}')
print(f'validation loss: {epoch_validation_losses[-1]}')

for e in range(epochs):
  sys.stdout.write(f'{e}:0 ')
  for i in range(minibatch_per_epoch):
      train_minibatch(boards_train_gpu, probs_train_gpu)
      if i % 10 == 9:
        sys.stdout.write('.')
        sys.stdout.flush()
  #if e == reduce_rate_at:
  #    print(f'reducing learning rate')
  #    optimizer.param_groups[0]['lr'] = 0.0005
  epoch_train_losses.append(evaluate_sample(boards_train_gpu, probs_train_gpu))
  epoch_validation_losses.append(evaluate_sample(boards_val_gpu, probs_val_gpu))
  print(f' | epoch {e}: training loss: {epoch_train_losses[-1]}, validation loss: {epoch_validation_losses[-1]}')

torch.save(action_model, './_out/model_2000r_100g.pt')
