import ctypes
import numpy.ctypeslib as ctl
import os

from numpy.ctypeslib import ndpointer

batch_mcts_lib = ctl.load_library("libmcts.so", os.path.join(
    os.path.dirname(__file__), "mctslib", "_build"))

LogFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int32)
BoolFn = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32)
EvalFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int32)
batch_mcts_lib.batch_mcts.argtypes = [
    ctypes.c_int, 
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    EvalFn,
    LogFn,
    BoolFn, # game_done_fn,
    ctypes.c_int, #model_a
    ctypes.c_int, #model_b
    ctypes.c_int,  # explore_for_n_moves
    ctypes.c_int32, # a_rollouts
    ctypes.c_double, # a_temp
    ctypes.c_int32, # b_rollouts
    ctypes.c_double # b_temp
]
batch_mcts_lib.batch_mcts.restype = None