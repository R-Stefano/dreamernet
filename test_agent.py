import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import shutil #clean folder for retraining
from utils import preprocessingState

from models import RNN, VAE, ACTOR
from trainer import Trainer
from EnvWrapper import EnvWrap

flags = tf.app.flags
FLAGS = flags.FLAGS
#ENVIRONMENT
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

#ACTOR
flags.DEFINE_boolean('training_ACTOR', True, 'If True, train the ACTOR model')
flags.DEFINE_integer('ACTOR_input_size', 192, 'THe dimension of input vector')

flags.DEFINE_integer('actor_training_games', 2, 'Number of games to play to generate the transitions to train the Actor')
flags.DEFINE_integer('actor_training_steps', 200, 'Number of training steps after the training games')
flags.DEFINE_integer('actor_training_epochs', 100, 'Number of times repeate games + steps')
flags.DEFINE_integer('actor_testing_games', 10, 'Number of games to play after each training')

vae_sess=tf.Session()
rnn_sess=tf.Session()
actor_sess=tf.Session()

env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env)
vae=VAE.VAE(vae_sess)
rnn=RNN.RNN(rnn_sess)
actor=ACTOR.ACTOR(actor_sess)
trainer=Trainer()

statesBuffer=[]
actionsBuffer=[]
rewardsBuffer=[]
hidden_state=[]
terminalBuffer=[]
for i in range(FLAGS.actor_training_games):
    s, d=env.initializeGame()
    h=np.zeros((1, 2, 1, FLAGS.hidden_units))
    game_rew=0
    while (not(d)):
        # quick state preprocessing
        s=preprocessingState(s)

        statesBuffer.append(s)
        hidden_state.append(h)

        #encode the state
        enc=vae.encode(s) #[1,64]
        
        evaluateInput=np.concatenate((enc, h[0,1]), axis=-1) #[1,192]
        policy, value=actor.predict(evaluateInput)

        a=np.argmax(policy)

        #predict next state but retrieve the encoded version #[1,128]
        exp_a=np.expand_dims(np.expand_dims(a, axis=0), axis=0)
        out,h=rnn.predict(np.expand_dims(np.concatenate((enc, exp_a), axis=-1), axis=1), initialize=h)

        h=np.expand_dims(np.asarray(h), axis=0)
        actionsBuffer.append(a)

        s, r, d=env.repeatStep(a)
        game_rew += r
        rewardsBuffer.append(r)
        terminalBuffer.append(d)

        if d:
            statesBuffer.append(np.zeros((statesBuffer[-1].shape)))
            actionsBuffer.append(a)
            rewardsBuffer.append(r)
            hidden_state.append(h)

    if len(statesBuffer)>FLAGS.sequence_length:
        
        #train vae using states
        idxs=np.random.randint(0, len(statesBuffer), 32)
        states=np.asarray(statesBuffer)[idxs]
        
        _, summ=vae.sess.run([vae.opt, vae.playing], feed_dict={vae.X: states})
        vae.file.add_summary(summ, i)
        vae.save()

        #train rnn using states, action, next state, reward
        inputData=np.zeros((FLAGS.RNN_train_size, FLAGS.sequence_length, FLAGS.latent_dimension+FLAGS.actions_size))
        labelData=np.zeros((FLAGS.RNN_train_size, FLAGS.sequence_length, FLAGS.latent_dimension+1))
        initStateData=np.zeros((1, 2, FLAGS.RNN_train_size, FLAGS.hidden_units))
        
        idxs=np.random.randint(FLAGS.sequence_length, len(statesBuffer)-1, 32)
        for j,idx in enumerate(idxs):
            start=idx-FLAGS.sequence_length
            #encode 101 states. Use first 100 for input and last 100 for output
            statesToEncode=np.asarray(statesBuffer[start:idx+1])
            encodedStates=vae.encode(statesToEncode)

            #retrieve the first 100 states and actions
            seqActions=np.expand_dims(np.asarray(actionsBuffer[start:idx]), axis=-1)
            #create Input data           
            inputData[j]=np.concatenate((encodedStates[:FLAGS.sequence_length], seqActions), axis=-1)

            seqRewards=np.expand_dims(np.asarray(rewardsBuffer[start:idx]), axis=-1)
            labelData[j]=np.concatenate((encodedStates[1:], seqRewards), axis=-1)

            initStateData[0,:,j,:]=np.asarray(hidden_state)[0,:,0]
        
        _, summ=rnn.sess.run([rnn.opt, rnn.playing], feed_dict={rnn.X: inputData,
                                                                rnn.true_next_state: labelData,
                                                                rnn.init_state: initStateData})
        rnn.file.add_summary(summ, i)
        rnn.save()
        
        #train actor with s,h and real reward
        #feed first states for vs1
        idxs=np.random.randint(0, len(statesBuffer)-1, 32)
        states=vae.encode(np.asarray(statesBuffer)[idxs])
        h=np.asarray(hidden_state)[idxs]
        h_state=np.squeeze(h[:,:,-1,:])
        vs1=actor.sess.run(actor.valueOutput, feed_dict={actor.X:np.concatenate((states, h_state), axis=-1)})
        
        #train the network
        idxs=idxs-1
        states=vae.encode(np.asarray(statesBuffer)[idxs])
        h=np.asarray(hidden_state)[idxs]
        h_state=np.squeeze(h[:,:,-1,:])

        #retrieve actions
        input_actions=np.expand_dims(np.asarray(actionsBuffer)[idxs], axis=-1)
        input_rewards=np.expand_dims(np.asarray(rewardsBuffer)[idxs], axis=-1)
        input_terminal=np.expand_dims(np.asarray(terminalBuffer)[idxs], axis=-1)

        print(input_terminal.shape)

        _, summ=actor.sess.run([actor.opt, actor.training], feed_dict={actor.X: np.concatenate((states, h_state), axis=-1),
                                               actor.actions: input_actions,
                                               actor.Vs1: vs1,
                                               actor.rewards: input_rewards,
                                               actor.isTerminal: input_terminal,
                                               actor.avgRew: game_rew})
        actor.file.add_summary(summ, i)
        actor.save()