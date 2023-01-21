import ctypes
import numpy as np
import numpy.ctypeslib as ctl
import os

from numpy.ctypeslib import ndpointer

class MNKLib:
  instance = None
  def __init__(self, lib, path):
    mnk = ctl.load_library(lib, path)

    mnk.new_state.argtypes = [ctypes.c_int]
    mnk.new_state.restype = ctypes.c_void_p

    mnk.destroy_state.argtypes = [ctypes.c_int, ctypes.c_void_p]
    mnk.destroy_state.restype = None

    mnk.new_mcts.argtypes = [ctypes.c_int]
    mnk.new_mcts.restype = ctypes.c_void_p

    mnk.destroy_mcts.argtypes = [ctypes.c_int, ctypes.c_void_p]
    mnk.destroy_mcts.restype = None

    mnk.state_apply.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    mnk.state_apply.restype = ctypes.c_bool

    mnk.state_finished.argtypes = [ctypes.c_int, ctypes.c_void_p]
    mnk.state_finished.restype = ctypes.c_bool

    mnk.state_winner.argtypes = [ctypes.c_int, ctypes.c_void_p]
    mnk.state_winner.restype = ctypes.c_int

    mnk.state_get_board.argtypes = [ctypes.c_int, ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
    mnk.state_get_board.restype = None

    mnk.state_get_boards.argtypes = [ctypes.c_int, ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
    mnk.state_get_boards.restype = None

    self.EvalFunction = ctypes.CFUNCTYPE(ctypes.c_void_p)

    mnk.mcts_get_moves.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), self.EvalFunction]
    mnk.mcts_get_moves.restype = None

    self.mnklib = mnk

  @classmethod
  def get(cls):
    if cls.instance == None:
      cls.instance = cls("libmnk.so", os.path.join(os.path.dirname(__file__), "mnklib", "_build"))
    return cls.instance

class State:
  def __init__(self, n):
    if n not in [6, 7, 8]:
      raise Exception("only 6, 7, 8 board size is supported")
    self.N = n;
    self.handle = MNKLib.get().mnklib.new_state(n)
      
  def __del__(self):
    MNKLib.get().mnklib.destroy_state(self.N, self.handle)

  def apply(self, move):
    # move is either tuple of numbers OR single index?
    # support tuple for now
    (x, y) = move
    return MNKLib.get().mnklib.state_apply(self.N, self.handle, x, y)

  def board(self):
    res = np.zeros(self.N * self.N, dtype=np.int32)
    MNKLib.get().mnklib.state_get_board(self.N, self.handle, res)
    return res.reshape(self.N, self.N)

  def boards(self):
    res = np.zeros(2 * self.N * self.N, dtype=np.int32)
    MNKLib.get().mnklib.state_get_boards(self.N, self.handle, res)
    return res.reshape(2, self.N, self.N)

  def finished(self):
    return MNKLib.get().mnklib.state_finished(self.N, self.handle)

  def winner(self):
    return MNKLib.get().mnklib.state_winner(self.N, self.handle)

# single instance of MCTS, not thread-safe. 1:1 relationship with thread
class MCTS:
  def __init__(self, n):
    if n not in [6, 7, 8]:
      raise Exception("only 6, 7, 8 board size is supported")
    self.N = n;
    self.handle = MNKLib.get().mnklib.new_mcts(n)

  def __del__(self):
    MNKLib.get().mnklib.destroy_mcts(self.N, self.handle)

  def run(self, state, temp, rollouts):
    def evalfn():
      pass
      #print("eval called")

    eval_function = MNKLib.get().EvalFunction(evalfn)
    moves = np.zeros(self.N * self.N, dtype=np.double)
    MNKLib.get().mnklib.mcts_get_moves(self.N, self.handle,
                                       state.handle, temp, rollouts, moves, eval_function)
    return moves.reshape(self.N, self.N)
