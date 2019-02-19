import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import shutil #clean folder for retraining
import os
from models import RNN, VAEGAN
from trainer import Trainer
from EnvWrapper import EnvWrap

test_envs={
    'frost':['FrostbiteNoFrameskip-v0', 18],
    'pong':['PongNoFrameskip-v0', 6]
}
flags = tf.app.flags
FLAGS = flags.FLAGS
#ENVIRONMENT #env basic is 210,160,3
flags.DEFINE_integer('img_size', 96, 'dimension of the state to feed into the VAE')
flags.DEFINE_integer('crop_size', 160, 'dimension of the state after crop')
flags.DEFINE_integer('actions_size', 1, 'Number of actions in the environment. box2d is 3, ataray games is 1')
flags.DEFINE_integer('num_actions', 18, 'Number of possible actions in the environment if action is discreate')
flags.DEFINE_integer('gap', 28, 'How much crop from the top of the image')
flags.DEFINE_integer('init_frame_skip', 30, 'Number of frames to skip at the beginning of each game')
flags.DEFINE_integer('frame_skip', 4, 'Number of times an action is repeated')
flags.DEFINE_string('env', 'FrostbiteNoFrameskip-v0', 'The environment to use') # AirRaidNoFrameskip-v0 # #BreakoutNoFrameskip-v0  #CrazyClimber #JourneyEscape #Tutankham
flags.DEFINE_integer('games', 5 , 'Number of times run the environment to create the data')
flags.DEFINE_boolean('renderGame', False , 'Set to True to render the game')

#GAN
flags.DEFINE_boolean('training_VAE', True, 'If True, train the VAE model')
flags.DEFINE_boolean('training_GAN', True, 'If True, train the GAN model')
flags.DEFINE_boolean('testing_VAEGAN', False, 'If true testing the VAEGAN')
flags.DEFINE_boolean('use_only_GAN_loss', True, 'If true the error for the generator is only the abiity to fool the discriminator, else it is also added the VAE error')
flags.DEFINE_float('weight_VAE_loss', 0.5, 'If use_only_GAN_loss is False, this value decide the weight ot give to the VAE loss on the generator')


flags.DEFINE_integer('GAN_epoches', 100, 'Nmber of times to repeat real-fake training')
flags.DEFINE_integer('GAN_disc_real_epoches', 50, 'Number of epoches to train the discriminator on real data')
flags.DEFINE_integer('GAN_disc_fake_epoches', 50, 'number of epoches to train the discriminator on fake data')
flags.DEFINE_integer('GAN_gen_epoches', 50, 'number of epoches to train the discriminator on fake data')
flags.DEFINE_integer('VAE_training_epoches', 50, 'Number of epoches to train VAE')

flags.DEFINE_integer('VAE_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('VAE_test_size', 64, 'Number of frames to feed at each epoch')
#VAE HYPERPARAMETERS
flags.DEFINE_integer('latent_dimension', 64, 'latent dimension')
flags.DEFINE_float('beta', 1, 'Disentangled Hyperparameter')

#RNN
flags.DEFINE_boolean('training_RNN', True, 'If True, train the RNN model')
flags.DEFINE_boolean('testing_RNN', False, 'If true testing the RNN')
flags.DEFINE_integer('RNN_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('RNN_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('RNN_test_size', 64, 'Number of frames to feed at each epoch')
#RNN HYPERPARAMETERS
flags.DEFINE_integer('sequence_length', 100, 'Total number of states to feed to the RNN')
flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in the LSTM layer')
flags.DEFINE_integer('LSTM_layers', 1, 'Number of the LSTM layers')

if(FLAGS.training_VAE and (len(os.listdir('models/VAEGAN/'))!=0) and FLAGS.training_GAN):
    print('cleaning VAE folder')
    shutil.rmtree('models/VAEGAN/')#clean folder
if(FLAGS.training_RNN and (len(os.listdir('models/RNN/'))!=0)):
    print('cleaning RNN folder')
    shutil.rmtree('models/RNN/')#clean folder

vae_sess=tf.Session()
rnn_sess=tf.Session()
env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env, FLAGS.renderGame)
vae=VAEGAN.VAEGAN(vae_sess)
rnn=RNN.RNN(rnn_sess)
trainer=Trainer()

frames, actions, rewards=env.run(FLAGS.games)

for i in frames[:10]:
    plt.imshow(i)
    plt.show()
#Training VAEGAN
trainer.prepareVAEGAN(frames.copy(), vae)

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

if (FLAGS.testing_VAEGAN):
    idxs=np.random.randint(0, frames.shape[0], 4)
    inputs=frames[idxs]
    
    out=vae.sess.run(vae.output, feed_dict={vae.X:inputs})#vae.decode(vae.encode(inputs))
    out=(out*255).astype(int)
    
    f, axarr = plt.subplots(4,2)
    for i in range(out.shape[0]):
        axarr[i,0].imshow(inputs[i])
        axarr[i,1].imshow(out[i])

    plt.show()