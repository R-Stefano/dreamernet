from utils import preprocessingState
import gym

class Env():
    def __init__(self):
        self.init_frame_skip=30
        self.frame_skip=4
        self.sampleGenerationEpochs=5
        self.statesBuffer=[]
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
            while (not(d)):
                if f_count % self.frame_skip == 0:
                    # quick state preprocessing
                    s=preprocessingState(s)
                    self.statesBuffer.append(s)
                    f_count=0
                a = self.env.action_space.sample()
                s, r, d, _ =self.env.step(a)
                f_count+=1
        
        return self.statesBuffer