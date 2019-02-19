from utils import preprocessingState
import gym
import numpy as np
import matplotlib.pyplot as plt
#1=NOOP
#2=UP
#3=DOWN

class EnvWrap():
    def __init__(self, init_skip, frame_skip, envName, renderGame):
        self.init_frame_skip=init_skip
        self.frame_skip=frame_skip
        self.statesBuffer=[]
        self.actionsBuffer=[]
        self.rewardsBuffer=[]
        self.env=gym.make(envName)
        self.renderGame=renderGame

    def run(self, simulation_epochs):
        for i in range(simulation_epochs):

            s, d=self.initializeGame()

            while (not(d)):
                if(self.renderGame):
                    self.env.render()
                #input('wait')
                # quick state preprocessing
                s=preprocessingState(s)
                self.statesBuffer.append(s)

                #randomly sample action 0,1,2
                a = self.env.action_space.sample()#np.random.randint(3)
                self.actionsBuffer.append(a)

                s, r, d=self.repeatStep(a) #a+1 for pong
                self.rewardsBuffer.append(r)

                if (d):
                    self.statesBuffer.append(np.zeros((self.statesBuffer[-1].shape)))
                    self.actionsBuffer.append(a)
                    self.rewardsBuffer.append(r)

        return np.asarray(self.statesBuffer).astype(int), np.asarray(self.actionsBuffer), np.asarray(self.rewardsBuffer)
    
    def initializeGame(self):
        s = self.env.reset()

        #wait that the environment is ready
        for i in range(self.init_frame_skip):
            a = self.env.action_space.sample()
            s, r, d, _ =self.env.step(a)
        
        return s, d
    
    def repeatStep(self, action):
        rew=0
        for i in range(self.frame_skip):
            s, r, d, _ =self.env.step(action)
            rew += r
        return s, rew, d
