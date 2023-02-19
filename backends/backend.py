def backend(device, torch_model, batch_size, board_size):
    if device in ['ane', 'mps']:
        from backends.backend_coreml import EvalBackend
        return EvalBackend(device, torch_model, batch_size, board_size)
    elif device.startswith('cuda'):
        from backends.backend_trt import EvalBackend
        return EvalBackend(device, torch_model, batch_size, board_size)
    else:
        from backends.backend_pytorch import EvalBackend
        return EvalBackend(device, torch_model, batch_size, board_size)