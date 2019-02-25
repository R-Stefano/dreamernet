import tensorflow as tf 
import numpy as np
import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env, vaegan, rnn, actor, trainer):
    print('Training actor..')
    statesBuffer=[]
    actionsBuffer=[]
    rewardsBuffer=[]
    h_statesBuffer=[]
    terminalBuffer=[]
    for game in range(FLAGS.actor_training_games):
        print('Actor playing game ({}/{})'.format(game, FLAGS.actor_training_games))
        s, d=env.initializeGame()
        lstmTuple=np.zeros((rnn.num_layers,2,1, rnn.hidden_units))
        h=lstmTuple[0,0]
        game_rew=0
        while (not(d)):
            # quick state preprocessing
            s=utils.preprocessingState(s)

            #Save data for system training
            statesBuffer.append(s)
            h_statesBuffer.append(lstmTuple)

            #encode the state
            enc=vaegan.encode(s) #[1,64]
            if (FLAGS.prediction_type == 'KL'):
                mu, std=np.split(enc, [rnn.latent_dimension], axis=-1)
                enc=mu + std*np.random.normal(size=rnn.latent_dimension)
            inputActor=np.concatenate((enc, h), axis=-1) #[1,192]
            policy, value=actor.predict(inputActor)

            if np.random.random()>0.2:
                a=np.argmax(policy)
            else:
                a=env.env.action_space.sample()

            #predict next state but retrieve the encoded version #[1,128]
            inputRNN=np.expand_dims(np.concatenate((enc, np.asarray(a).reshape((1,1))), axis=-1), axis=1)
            _,lstmTuple=rnn.predict(inputRNN, initialize=lstmTuple) #lstmTuple=[1,2,1,128] where 2=hidden,cell

            h=lstmTuple[0,0]
            actionsBuffer.append(a)

            s, r, d=env.repeatStep(a)
            print(a)
            game_rew += r
            rewardsBuffer.append(r)
            terminalBuffer.append(d)

            #keep max size of FLAGS.transition_buffer_size transitions
            if (len(statesBuffer) > FLAGS.transition_buffer_size):
                del statesBuffer[0]
                del actionsBuffer[0]
                del rewardsBuffer[0]
                del h_statesBuffer[0]
                del terminalBuffer[0]

            if d:
                statesBuffer.append(np.zeros((statesBuffer[-1].shape)))
                actionsBuffer.append(a)
                rewardsBuffer.append(r)
                h_statesBuffer.append(lstmTuple)
                terminalBuffer.append(d)


        if len(statesBuffer)>FLAGS.sequence_length:
            print('Training system..')
            trainer.trainSystem(np.asarray(statesBuffer),
                                np.asarray(actionsBuffer),
                                np.asarray(rewardsBuffer),
                                np.asarray(terminalBuffer), 
                                np.asarray(h_statesBuffer),
                                vaegan, 
                                rnn,
                                actor,
                                game)
            
            summ=actor.sess.run(actor.game, feed_dict={actor.avgRew: game_rew})
            actor.file.add_summary(summ, game)


        
