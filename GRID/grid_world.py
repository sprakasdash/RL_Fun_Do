from __future__ import print_function, division
from builtins import range
import numpy as np

class Grid: # Environment
  def __init__(self, rows, cols, start):
    self.rows = rows
    self.cols = cols
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'U':
        self.i -= 1
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
    if action == 'U':
      self.i += 1
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state() in self.all_states())

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    
    return set(self.actions.keys()) | set(self.rewards.keys())

def euclid_dis(x, y):
    return (x**2 + y**2)**0.5

def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # $ means highest reward position
  # one of the state has FIRE 
  # s   .   .   .
  # .   x   . (FIRE)
  # .   .   .  ($)
  g = Grid(3, 4, (0, 0))
  rewards = {
    (0, 0): -1 + euclid_dis(0, 0),
    (0, 1): -1 + euclid_dis(0, 1),
    (0, 2): -1 + euclid_dis(0, 2),
    (0, 3): -1 + euclid_dis(0, 3), 
    (1, 0): -1 + euclid_dis(1, 0),
    (1, 2): -1 + euclid_dis(1, 2),
    (1, 3): -1, #fire, can't go there
    (2, 0): -1 + euclid_dis(2, 0),
    (2, 1): -1 + euclid_dis(2, 1),
    (2, 2): -1 + euclid_dis(2, 2),
    (2, 3): -1 + euclid_dis(2, 3),
  }
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }
  g.set(rewards, actions)
  return g