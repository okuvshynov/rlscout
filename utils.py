import matplotlib.pyplot as plt
import torch
import io
from action_value_model import ActionValueModel as ActionNN
import coremltools as ct

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

def save_sample(db, board_tensor, probs_tensor, model_id):
  for b, p in zip(symm(board_tensor), symm(probs_tensor.view(1, 8, 8))):
    board_buffer = io.BytesIO()
    torch.save(b, board_buffer)
    prob_buffer = io.BytesIO()
    torch.save(p, prob_buffer)
    db.append_sample(board_buffer.getvalue(), prob_buffer.getvalue(), model_id)

def to_coreml(torch_model):
    torch_model.eval()
    sample = torch.rand(1, 2, 8, 8)

    traced_model = torch.jit.trace(torch_model, sample)
    return ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)],
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )