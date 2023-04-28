import torch

def pick_device():
    if torch.backends.mps.is_available():
       return "ane"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
    
def pick_train_device():
    if torch.backends.mps.is_available():
       return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"