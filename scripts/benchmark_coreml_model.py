import coremltools as ct
import numpy as np
import time
import torch
from fake_model import ActionValueModel

cpu, gpu, ne = False, False, True

def to_coreml(torch_model, batch_size, compute_units):
    torch_model = torch_model.cpu()
    torch_model.eval()
    sample = torch.rand(batch_size, 2, 8, 8)

    traced_model = torch.jit.trace(torch_model, sample)
    return ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)],
        compute_units=compute_units
    )

model = ActionValueModel()

for log_batch_size in range(10):
    batch_size = 2 ** log_batch_size
    sample = {'x': np.random.rand(batch_size, 2, 8, 8)}
    repeat = 50000
    if cpu:
        cpu_model = to_coreml(model, batch_size, compute_units=ct.ComputeUnit.CPU_ONLY)
        start = time.time()
        for rep in range(repeat):
            out = cpu_model.predict(sample)

        print(f'cpu: 1k x {batch_size} {time.time() - start}')

    if gpu:
        gpu_model = to_coreml(model, batch_size, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        start = time.time()
        for rep in range(repeat):
            out = gpu_model.predict(sample)

        print(f'gpu: 1k x {batch_size} {time.time() - start}')
    
    if ne:
        ne_model = to_coreml(model, batch_size, compute_units=ct.ComputeUnit.CPU_AND_NE)
        start = time.time()
        for rep in range(repeat):
            out = ne_model.predict(sample)

        print(f'ne: 1k x {batch_size} {time.time() - start}')
