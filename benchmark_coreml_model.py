import coremltools as ct
import numpy as np
import time

cpu, gpu, ne = True, True, True

for log_batch_size in range(4):
    batch_size = 2 ** log_batch_size
    sample = {'x': np.random.rand(batch_size, 2, 8, 8)}
    if cpu:
        cpu_model = ct.models.MLModel(f'./_out/8x8/coreml_model_cp_{batch_size}.mlmodel', compute_units=ct.ComputeUnit.CPU_ONLY)
        start = time.time()
        for rep in range(10000):
            out = cpu_model.predict(sample)

        print(f'cpu: 10k x {batch_size} {time.time() - start}')

    if gpu:
        gpu_model = ct.models.MLModel(f'./_out/8x8/coreml_model_cp_{batch_size}.mlmodel', compute_units=ct.ComputeUnit.CPU_AND_GPU)
        start = time.time()
        for rep in range(10000):
            out = gpu_model.predict(sample)

        print(f'gpu: 10k x {batch_size} {time.time() - start}')
    
    if ne:
        ne_model = ct.models.MLModel(f'./_out/8x8/coreml_model_cp_{batch_size}.mlmodel', compute_units=ct.ComputeUnit.CPU_AND_NE)
        start = time.time()
        for rep in range(10000):
            out = ne_model.predict(sample)

        print(f'ne: 10k x {batch_size} {time.time() - start}')