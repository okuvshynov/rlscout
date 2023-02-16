def backend(device, torch_model, batch_size, board_size):
    if device in ['ane', 'mps']:
        from backend_coreml import EvalBackend
        return EvalBackend(device, torch_model, batch_size, board_size)
    elif device.startswith('cuda'):
        from backend_trt import EvalBackend
        return EvalBackend(device, torch_model, batch_size, board_size)
    else:
        from backend_pytorch import EvalBackend
        return EvalBackend(device, torch_model, batch_size, board_size)