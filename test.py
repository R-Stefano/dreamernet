from MCTS import Tree
import numpy as np

'''
OKay this is my environment.
At every timestep, the env return a state.

I need that MCTS, given the state, return to me
what action I should do
'''
num_actions=3
mcts=Tree(num_actions)

def env():
    return np.random.randint(0,255,(64,64))

s=env()

a=mcts.predict(s)
