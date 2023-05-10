import ctypes
import numpy.ctypeslib as ctl
import os
import logging

from numpy.ctypeslib import ndpointer
from utils.utils import random_seed

rlslib = ctl.load_library("librls.so", os.path.join(
    os.path.dirname(__file__), "_build"))

LogFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64, ctypes.c_int8, ctypes.c_int8)
GameDoneFn = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32, ctypes.c_int64)
EvalFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool)
rlslib.batch_mcts.argtypes = [
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
rlslib.batch_mcts.restype = None

PyLogFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
rlslib.init_py_logger.argtypes = [PyLogFn]
rlslib.init_py_logger.restype = None

def py_log_impl_cb(level, msg):
    level = level.decode('utf-8')
    msg = msg.decode('utf-8')
    if level == 'info':
        logging.info(msg)

py_log_fn = PyLogFn(py_log_impl_cb)

rlslib.init_py_logger(py_log_fn)

rlslib.init_random_seed.argtypes = [ctypes.c_int64]
rlslib.init_random_seed.restype = None

rlslib.init_random_seed(random_seed())

rlslib.ab_duel.argtypes = [
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
rlslib.ab_duel.restype = None

rlslib.run_ab.restype = ctypes.c_int8
rlslib.run_ab.argtypes = [
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # boards
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # probs
    EvalFn,
    ctypes.c_int8, # alpha
    ctypes.c_int8, # beta
]