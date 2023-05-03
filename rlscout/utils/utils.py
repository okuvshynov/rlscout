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

# takes a string representing list of ids.
# can be comma-separated list of non-negative integers or ranges [a-b].
# examples:
# 1
# 1,2,3,10
# 1-5,10-15,20
# repetitions are allowed
# order is preserved
def parse_ids(id_list):
    if not id_list:
        return []
    res = []
    make_range = lambda ts: range(int(ts[0]), int(ts[-1]) + 1)
    for token in id_list.split(','):
        res.extend(make_range(token.split('-')))

    return res