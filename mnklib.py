import ctypes
import numpy as np
import numpy.ctypeslib as ctl
import os
import sys

from numpy.ctypeslib import ndpointer

class MNKLib:
  instance = None
  def __init__(self, lib, path):
    mnk = ctl.load_library(lib, path)

    mnk.new_state.argtypes = [ctypes.c_int]
    mnk.new_state.restype = ctypes.c_void_p

    mnk.destroy_state.argtypes = [ctypes.c_int, ctypes.c_void_p]
    mnk.destroy_state.restype = None

    # two buffers - IN/OUT for model evaluation.
    mnk.new_mcts.argtypes = [ctypes.c_int, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
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

  # player is 0 or 1
  def pp(self, player, last_move=None, f=sys.stdout):
    chr_player = {
        (-1, False): '.',
        (1-player, True): '0',
        (1-player, False): 'o',
        (player, True): 'X',
        (player, False): 'x'
    }
    board = self.board()
    for i in range(self.N):
      f.write(os.linesep)
      for j in range(self.N):
        f.write(chr_player[(board[i, j], (i, j) == last_move)])
    f.flush()

# single instance of MCTS, not thread-safe. 1:1 relationship with thread
class MCTS:
  def __init__(self, n):
    if n not in [6, 7, 8]:
      raise Exception("only 6, 7, 8 board size is supported")
    self.N = n;
    self.boards_buffer = np.zeros(2 * self.N * self.N, dtype=np.int32)
    self.probs_buffer = np.ones(self.N * self.N, dtype=np.float32)
    self.handle = MNKLib.get().mnklib.new_mcts(n, self.boards_buffer, self.probs_buffer)

  def __del__(self):
    MNKLib.get().mnklib.destroy_mcts(self.N, self.handle)

  def run(self, state, temp, rollouts, get_probs_fn=None):
    def eval_fn():
      get_probs_fn(self.boards_buffer, self.probs_buffer)

    if get_probs_fn is None:
      eval_function = MNKLib.get().EvalFunction(0)
    else:
      eval_function = MNKLib.get().EvalFunction(eval_fn)

    moves = np.zeros(self.N * self.N, dtype=np.double)
    MNKLib.get().mnklib.mcts_get_moves(self.N, self.handle,
                                       state.handle, temp, rollouts, moves, eval_function)
    return moves.reshape(self.N, self.N)