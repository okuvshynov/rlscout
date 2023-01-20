import coremltools as ct
import numpy as np
import time

#print(cpu_model.get_spec().description.input)
cpu, gpu, ne = False, False, True

for log_batch_size in range(10):
    sample = {'x': np.random.rand(2 ** log_batch_size, 2, 7, 7)}
    if cpu:
        cpu_model = ct.models.MLModel(f'./_out/coreml_model_cp_{2 ** log_batch_size}.mlmodel', compute_units=ct.ComputeUnit.CPU_ONLY)
        start = time.time()
        for rep in range(10000):
            out = cpu_model.predict(sample)

        print(f'cpu: 1k x {2 ** log_batch_size} {time.time() - start}')

    if gpu:
        gpu_model = ct.models.MLModel(f'./_out/coreml_model_cp_{2 ** log_batch_size}.mlmodel', compute_units=ct.ComputeUnit.CPU_AND_GPU)
        start = time.time()
        for rep in range(10000):
            out = gpu_model.predict(sample)

        print(f'gpu: 1k x {2 ** log_batch_size} {time.time() - start}')
    
    if ne:
        ne_model = ct.models.MLModel(f'./_out/coreml_model_cp_{2 ** log_batch_size}.mlmodel', compute_units=ct.ComputeUnit.CPU_AND_NE)
        start = time.time()
        for rep in range(10000):
            out = ne_model.predict(sample)

        print(f'ne: 1k x {2 ** log_batch_size} {time.time() - start}')