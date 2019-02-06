from utils import preprocessingState
import gym
import numpy as np

class Env():
    def __init__(self):
        self.init_frame_skip=30
        self.frame_skip=4
        self.sampleGenerationEpochs=1
        self.statesBuffer=[]
        self.actionsBuffer=[]
        self.env=gym.make('PongNoFrameskip-v4')

    def run(self):
        for i in range(self.sampleGenerationEpochs):
            s = self.env.reset()
            d = False

            #wait that the environment is ready
            for i in range(self.init_frame_skip):
                a = self.env.action_space.sample()
                s, r, d, _ =self.env.step(a)

            f_count = 0
            rew=0
            while (not(d)):
                if f_count % self.frame_skip == 0:
                    # quick state preprocessing
                    s=preprocessingState(s)
                    self.statesBuffer.append(s)
                    a = self.env.action_space.sample()
                    self.actionsBuffer.append(a)
                    rew=0
                    f_count=0
                else:
                    f_count+=1
                
                s, r, d, _ =self.env.step(a)
                rew += r 
                if d:
                    self.statesBuffer.append(np.zeros((64,64,3)))
                    self.actionsBuffer.append(-1)
            
            print(len(self.statesBuffer))
            print(len(self.actionsBuffer))
                
        
        return self.statesBuffer, self.actionsBuffer

env=Env()
env.run()