import ctypes
import numpy.ctypeslib as ctl
import os
import logging

from numpy.ctypeslib import ndpointer
from utils.utils import random_seed

LogFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64, ctypes.c_int8, ctypes.c_int8)
GameDoneFn = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32, ctypes.c_int64)
EvalFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool)
ModelIDFn = ctypes.CFUNCTYPE(ctypes.c_uint32)

class RLScoutNative:
    def __init__(self, seed=random_seed()) -> None:
        self.seed = seed
        self.lib = ctl.load_library("librls.so", os.path.join(
            os.path.dirname(__file__), "_build"))
        
        # batch_mcts API
        self.lib.batch_mcts.argtypes = [
            ctypes.c_int, 
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # boards
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # probs
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # log boards
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # log scores
            EvalFn,
            LogFn,
            GameDoneFn, # game_done_fn,
            ModelIDFn, #model_a
            ModelIDFn, #model_b
            ctypes.c_int,  # explore_for_n_moves
            ctypes.c_int32, # a_rollouts
            ctypes.c_double, # a_temp
            ctypes.c_int32, # b_rollouts
            ctypes.c_double, # b_temp
            ctypes.c_uint32, # a random rollouts
            ctypes.c_uint32, # b random rollouts
        ]
        self.lib.batch_mcts.restype = None

        # Run AB API
        self.lib.run_ab.restype = ctypes.c_uint64
        self.lib.run_ab.argtypes = [
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),   # boards
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # probs
            EvalFn,
            ctypes.c_int8, # alpha
            ctypes.c_int8, # beta
        ]

        PyLogFn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
        self.lib.init_py_logger.argtypes = [PyLogFn]
        self.lib.init_py_logger.restype = None
        def py_log_impl_cb(level, msg):
            level = level.decode('utf-8')
            msg = msg.decode('utf-8')
            if level == 'info':
                logging.info(msg)
        self.py_log_fn = PyLogFn(py_log_impl_cb)
        self.lib.init_py_logger(self.py_log_fn)

        self.lib.init_random_seed.argtypes = [ctypes.c_int64]
        self.lib.init_random_seed.restype = None
        self.lib.init_random_seed(self.seed)