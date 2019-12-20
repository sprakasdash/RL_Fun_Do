from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import standard_grid

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def print_values(V, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="")
    print("")

def print_policy(P, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")

if __name__ == '__main__':
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

  # initial policy
  print("initial policy:")
  print_policy(policy, grid)

  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    # V[s] = 0
    if s in grid.actions:
      V[s] = np.random.random()
    else:
      # terminal state
      V[s] = 0

  # repeat until convergence - will break out when policy does not change
  while True:
    # policy evaluation step
    while True:
      biggest_change = 0
      for s in states:
        old_v = V[s]

        # V(s) only has value if it's not a terminal state
        if s in policy:
          a = policy[s]
          grid.set_state(s)
          r = grid.move(a)
          V[s] = r + GAMMA * V[grid.current_state()]
          biggest_change = max(biggest_change, np.abs(old_v - V[s]))

      if biggest_change < SMALL_ENOUGH:
        break

    # policy improvement step
    is_policy_converged = True
    for s in states:
      if s in policy:
        old_a = policy[s]
        new_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in ALL_POSSIBLE_ACTIONS:
          grid.set_state(s)
          r = grid.move(a)
          v = r + GAMMA * V[grid.current_state()]
          if v > best_value:
            best_value = v
            new_a = a
        policy[s] = new_a
        if new_a != old_a:
          is_policy_converged = False

    if is_policy_converged:
      break

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)