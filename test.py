import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import shutil #clean folder for retraining
import os
from models import RNN, VAE
from trainer import Trainer
from EnvWrapper import EnvWrap

flags = tf.app.flags
FLAGS = flags.FLAGS
#ENVIRONMENT #env basic is 210,160,3
flags.DEFINE_integer('img_size', 96, 'dimension of the state to feed into the VAE')
flags.DEFINE_integer('crop_size', 160, 'dimension of the state after crop')
flags.DEFINE_integer('actions_size', 1, 'Number of actions in the environment. box2d is 3, ataray games is 1')
flags.DEFINE_integer('num_actions', 6, 'Number of possible actions in the environment if action is discreate')
flags.DEFINE_integer('gap', 35, 'How much crop from the top of the image')
flags.DEFINE_integer('init_frame_skip', 30, 'Number of frames to skip at the beginning of each game')
flags.DEFINE_integer('frame_skip', 4, 'Number of times an action is repeated')
flags.DEFINE_string('env', 'PongNoFrameskip-v0', 'The environment to use') #AirRaidNoFrameskip-v0 # #BreakoutNoFrameskip-v0
flags.DEFINE_integer('games', 3 , 'Number of times run the environment to create the data')

#VAE
flags.DEFINE_boolean('training_VAE', False, 'If True, train the VAE model')
flags.DEFINE_boolean('testing_VAE', False, 'If true testing the VAE')
flags.DEFINE_integer('VAE_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('VAE_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('VAE_test_size', 64, 'Number of frames to feed at each epoch')
#VAE HYPERPARAMETERS
flags.DEFINE_integer('latent_dimension', 64, 'latent dimension')
flags.DEFINE_float('beta', 1, 'Disentangled Hyperparameter')

#RNN
flags.DEFINE_boolean('training_RNN', False, 'If True, train the RNN model')
flags.DEFINE_boolean('testing_RNN', False, 'If true testing the RNN')
flags.DEFINE_integer('RNN_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('RNN_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('RNN_test_size', 64, 'Number of frames to feed at each epoch')
#RNN HYPERPARAMETERS
flags.DEFINE_integer('sequence_length', 100, 'Total number of states to feed to the RNN')
flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in the LSTM layer')
flags.DEFINE_integer('LSTM_layers', 1, 'Number of the LSTM layers')

if(FLAGS.training_VAE and (len(os.listdir('models/VAE/'))!=0)):
    print('cleaning VAE folder')
    shutil.rmtree('models/VAE/')#clean folder
if(FLAGS.training_RNN and (len(os.listdir('models/RNN/'))!=0)):
    print('cleaning RNN folder')
    shutil.rmtree('models/RNN/')#clean folder

vae_sess=tf.Session()
rnn_sess=tf.Session()
env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env)
vae=VAE.VAE(vae_sess)
rnn=RNN.RNN(rnn_sess)
trainer=Trainer()

frames, actions, rewards=env.run(FLAGS.games)

#frames=frames[:10]
#for f in frames:
#    plt.imshow(f)
#    plt.show()

#Training VAE
if(FLAGS.training_VAE):
    trainer.prepareVAE(frames.copy(), vae)

#Training RNN
embeds=vae.encode(frames)
if(FLAGS.training_RNN):
    trainer.prepareRNN(embeds, actions, rewards, rnn)

if(FLAGS.testing_RNN):
    idxs=np.random.randint(FLAGS.sequence_length, embeds.shape[0], 10)
    errors=[]
    for idx in idxs:
        sequenceEmbeds=embeds[(idx-FLAGS.sequence_length):idx]
        actionLength=np.expand_dims(actions[(idx-FLAGS.sequence_length):idx], axis=-1)
        inputData=np.concatenate((sequenceEmbeds, actionLength), axis=-1)
        out, h=rnn.predict(np.expand_dims(inputData, axis=0))
        #These states are going to be reconstruct
        inputEmbed=(sequenceEmbeds[-2:]*255).astype(int)
        outputEmbed=(out[-2:, :-1]*255).astype(int)

        reconstructInputState=vae.decode(inputEmbed)
        reconstructOutState=vae.decode(outputEmbed)
        f, axarr = plt.subplots(2,2)
        for i in range(reconstructInputState.shape[0]):
            axarr[i,0].imshow((reconstructInputState[i]*255).astype(int))
            axarr[i,1].imshow((reconstructOutState[i]*255).astype(int))

        plt.show()
        #check the error difference
        err=out[-1,-1] - rewards[idx]
        errors.append(err)
    
    print('avg error', np.mean(errors))

if (FLAGS.testing_VAE):
    idxs=np.random.randint(0, frames.shape[0], 4)
    inputs=frames[idxs]
    
    out=vae.sess.run(vae.output, feed_dict={vae.X:inputs})#vae.decode(vae.encode(inputs))
    out=(out*255).astype(int)
    
    f, axarr = plt.subplots(4,2)
    for i in range(out.shape[0]):
        axarr[i,0].imshow(inputs[i])
        axarr[i,1].imshow(out[i])

    plt.show()
#Training the actor