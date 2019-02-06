from MCTS import Tree
import numpy as np
import gym

'''
OKay this is my environment.
At every timestep, the env return a state.

I need that MCTS, given the state, return to me
what action I should do

num_actions=3
rollouts=100
mcts=Tree(num_actions, rollouts)

def env():
    return np.random.randint(0,255,(64,64))

s=env()

a=mcts.predict(s)
print(a)
'''

env=gym.make('PongNoFrameskip-v4')
env.reset()
#wait that the environment is ready
for i in range(30):
    a = env.action_space.sample()
    s, r, d, _ =env.step(a)
d=False
while not(d):
    env.render()
    a = np.random.randint(3)
    print(a)
    for i in range(4):
        env.step(a+1)
#0=NOOP
#2=UP
#3=DOWN