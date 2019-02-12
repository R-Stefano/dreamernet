from MCTS import Tree
import numpy as np
import gym
import tensorflow as tf 
from utils import preprocessingState
from VAE import VAE
from RNN import RNN
from decoder import Decoder
import matplotlib.pyplot as plt 

sess=tf.Session()
vae = VAE(sess, isTraining=False)
rnn=RNN(sess, isTraining=False)
#vaePredict = Decoder(sess)


env=gym.make('PongNoFrameskip-v4')
env.reset()
states=[]
actions=[]
embeddings=[]

num_actions=3
#wait that the environment is ready
for i in range(30):
    a = env.action_space.sample()
    s, r, d, _ =env.step(a)
d=False
while not(d):
    a = np.random.randint(num_actions)
    states.append(s)
    actions.append(a)
    embeddings.append(vae.predict(preprocessingState(s)))
    
    for i in range(4):
        s, r, d, _ =env.step(a+1)

idx=np.random.randint(0, len(states)-1)
currentState=states[idx]
trueNext=states[idx+1]

currentEmbed=embeddings[idx]
#prepare input RNN

#initialize hidden state and cell state to zeros 
cell_s=np.zeros((1, rnn.hidden_units))
hidden_s=np.zeros((1, rnn.hidden_units))

sequenceStates=np.squeeze(np.asarray(embeddings[idx-9:idx+1]))
actionSequence=np.expand_dims(np.asarray(actions[idx-9:idx+1]), axis=-1)
inputData=np.concatenate((sequenceStates, actionSequence), axis=-1)


nextEmbedPredicted=rnn.predict(np.expand_dims(inputData, axis=0), cell_s, hidden_s)
nextEmbedVAE=embeddings[idx+1]

resultPred=vae.embedDecod(nextEmbedPredicted)
resultEncodded=vae.embedDecod(nextEmbedVAE)

plt.imshow(np.squeeze(resultPred))
plt.show()
plt.imshow(np.squeeze(resultEncodded))
plt.show()
print(nextEmbedPredicted)
print(nextEmbedVAE)


