import ctypes
import numpy.ctypeslib as ctl
import os

from numpy.ctypeslib import ndpointer

batch_mcts_lib = ctl.load_library("libmcts.so", os.path.join(
    os.path.dirname(__file__), "..", "mctslib", "_build"))

batch_duel_lib = ctl.load_library("libduel.so", os.path.join(
    os.path.dirname(__file__), "..", "mctslib", "_build"))

LogFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64, ctypes.c_int8, ctypes.c_int8)
GameDoneFn = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32, ctypes.c_int64)
EvalFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool)
batch_mcts_lib.batch_mcts.argtypes = [
    ctypes.c_int, 
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # boards
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # probs
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # log boards
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # log scores
    EvalFn,
    LogFn,
    GameDoneFn, # game_done_fn,
    ctypes.c_int, #model_a
    ctypes.c_int, #model_b
    ctypes.c_int,  # explore_for_n_moves
    ctypes.c_int32, # a_rollouts
    ctypes.c_double, # a_temp
    ctypes.c_int32, # b_rollouts
    ctypes.c_double, # b_temp
    ctypes.c_uint32, # a random rollouts
    ctypes.c_uint32, # b random rollouts
]
batch_mcts_lib.batch_mcts.restype = None

batch_duel_lib.ab_duel.argtypes = [
    ctypes.c_int, 
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # boards
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # probs
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores
    EvalFn,
    GameDoneFn, # game_done_fn,
    ctypes.c_int, #model_id
    ctypes.c_int,  # explore_for_n_moves
    ctypes.c_int32, # rollouts
    ctypes.c_double, # temp
    ctypes.c_int8, # alpha
    ctypes.c_int8, # beta
    ctypes.c_uint32, # full_after_n_moves
    ctypes.c_bool, # inverse player order
    ctypes.c_uint32, # random rollouts at leaf
]
batch_duel_lib.ab_duel.restype = None