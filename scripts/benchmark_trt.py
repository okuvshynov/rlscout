from torch2trt import torch2trt
import numpy as np
import time
import torch
from fake_model import ActionValueModel

def to_trt(torch_model, batch_size, device):
    torch_model = torch_model.eval().to(device)
    sample = torch.rand(batch_size, 2, 8, 8).detach().to(device)
    return torch2trt(torch_model, [sample])

run_for_nseconds = 30
step_target = 0.05 * run_for_nseconds
device = 'cuda'

for nblocks in range(1, 15):
    model = ActionValueModel(nblocks=nblocks)
    for log_batch_size in range(12):
        batch_size = 2 ** log_batch_size

        sample = torch.rand(batch_size, 2, 8, 8).detach().to(device)
        
        ne_model = to_trt(model, batch_size, device)

        start = time.time()
        it = 0
        step = 100
        while True:
            for _ in range(step):
                out = ne_model(sample)
            it += step
            curr = time.time()
            if curr > run_for_nseconds + start:
                break
            if curr < start + step_target:
                step *= 2
        
        duration = time.time() - start
        total_ranked = it * batch_size
        ms_per_sample = 1000.0 * duration / total_ranked

        print(f'{nblocks},{batch_size},{duration:.3f},{total_ranked},{ms_per_sample:.3f}')
