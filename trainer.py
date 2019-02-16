import tensorflow as tf
import numpy as np
from utils import preprocessingState

flags = tf.app.flags
FLAGS = flags.FLAGS

class Trainer():
    def trainVAE(self,frames, vae):
        print('Starting VAE training..')
        training_epoches=FLAGS.VAE_training_epoches
        train_batch_size=FLAGS.VAE_train_size
        test_batch_size=FLAGS.VAE_test_size

        train_dataset=frames[:int(len(frames)*0.75)]
        test_dataset=frames[int(len(frames)*0.75):]

        for ep in range(training_epoches):
            print('Training VAE, epoch ({}/{})'.format(ep,training_epoches))

            #Sample from frames generated
            idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
            batchData=train_dataset[idxs]

            _, summ = vae.sess.run([vae.opt, vae.training], feed_dict={vae.X: batchData})

            vae.file.add_summary(summ, ep)

            if ep % 50 ==0:
                print('Saving VAE..')
                vae.save()

                rec_loss=0
                kl_loss=0
                tot_loss=0
                avg =0
                print('Testing VAE..')
                for batchStart in range(0, len(test_dataset), test_batch_size):
                    batchEnd=batchStart+test_batch_size
                    batchData=test_dataset[batchStart:batchEnd]

                    l1, l2, l3= vae.sess.run([vae.reconstr_loss, vae.KLLoss, vae.totLoss], feed_dict={vae.X: batchData})

                    rec_loss +=l1
                    kl_loss += l2
                    tot_loss += l3
                    avg +=1

                summ = vae.sess.run(vae.testTensorboard, feed_dict={vae.recLossPlace: (rec_loss/avg), vae.klLossPlace: (kl_loss/avg), vae.totLossPlace: (tot_loss/avg)})
                vae.file.add_summary(summ, ep)

    def trainRNN(self,frames, actions, rewards, rnn):
        print('Starting RNN training..')
        training_epoches=FLAGS.RNN_training_epoches
        train_batch_size=FLAGS.RNN_train_size
        test_batch_size=FLAGS.RNN_test_size

        train_dataset=(frames[:int(len(frames)*0.75)], actions[:int(len(frames)*0.75)], rewards[:int(len(frames)*0.75)])
        test_dataset=(frames[int(len(frames)*0.75):], actions[int(len(frames)*0.75):], rewards[int(len(frames)*0.75):])

        for ep in range(training_epoches):
            print('Training RNN, epoch ({}/{})'.format(ep,training_epoches))
            inputData, labelData = self.prepareRNNData(train_batch_size, rnn.sequence_length, rnn.latent_dimension, train_dataset)
            
            #initialize hidden state and cell state to zeros 
            init_state=np.zeros((rnn.num_layers, 2, train_batch_size, rnn.hidden_units))

            #Train
            _, summ = rnn.sess.run([rnn.opt, rnn.training], feed_dict={rnn.X: inputData, 
                                                         rnn.true_next_state: labelData,
                                                         rnn.init_state: init_state})
            rnn.file.add_summary(summ, ep)
            #Saving and testing
            if ep % 50 ==0:
                print('Saving RNN..')
                rnn.save()

                totLoss=0
                avg=0
                print('Testing RNN..')
                for b in range(len(test_dataset[0])//test_batch_size):
                    inputData, labelData = self.prepareRNNData(test_batch_size, rnn.sequence_length, rnn.latent_dimension, test_dataset)

                    #initialize hidden state and cell state to zeros 
                    init_state=np.zeros((rnn.num_layers, 2, test_batch_size, rnn.hidden_units))

                    #Train
                    loss = rnn.sess.run([rnn.loss], feed_dict={rnn.X: inputData, 
                                                           rnn.true_next_state: labelData,
                                                           rnn.init_state: init_state})
                    totLoss += loss[0]
                    avg +=1

                summ = rnn.sess.run(rnn.testing, feed_dict={rnn.totLossPlace: (totLoss/avg)})
                rnn.file.add_summary(summ, ep)

    #Called to train alphazero using everything.
    def trainActor(self,mcts, vae, rnn, env, actor):
        training_games=FLAGS.actor_training_games
        epochs=FLAGS.actor_training_epochs
        training_steps=FLAGS.actor_training_steps
        testing_games=FLAGS.actor_testing_games

        step=0
        for ep in range(epochs):
            print('Training Actor, epoch ({}/{})'.format(ep,epochs))

            print('Generating games..')
            states, actions, rewards, isTerminal=self.play(training_games, env, vae, mcts)

            print('training on sampled transitions..')
            for tr in range(training_steps):
                batchNextStates=[]
                batchStates=[]
                batchActions=[]
                batchRewards=[]
                batchTerminal=[]
                #Generate a batch of 32 examples to feed to train the network
                for i in range(32):
                    idx=np.random.randint(0, len(states)-1)
                    batchNextStates.append(states[idx+1])
                    batchStates.append(states[idx])
                    batchActions.append(actions[idx])
                    batchRewards.append(rewards[idx])
                    batchTerminal.append(isTerminal[idx])
                
                batchTerminal=np.asarray(batchTerminal).astype(int)
                #First feed the next state to the network to obtain the prediction about the value of V(s')
                _, s1Values=actor.predict(np.asarray(batchNextStates)) #return batch_size,1
                _, summ,=actor.sess.run([actor.opt, actor.training], feed_dict={actor.X: np.asarray(batchStates),
                                                                                actor.rewards: np.asarray(np.expand_dims(batchRewards, axis=-1)),
                                                                                actor.actions: np.asarray(batchActions),
                                                                                actor.Vs1: s1Values,
                                                                                actor.isTerminal: batchTerminal})
                actor.file.add_summary(summ, step)
                step+=1
            print('Saving the actor..')
            actor.save() 

            print('Testing the actor..')
            avg_rew=[]
            for i in range(testing_games):
                _, _, rewards, _=self.play(1, env, vae, mcts)
                avg_rew.append(np.sum(rewards))
            
            print('avg test rew', np.mean(avg_rew))
            summ=actor.sess.run(actor.testing, feed_dict={actor.avgRew: np.mean(avg_rew)})
            actor.file.add_summary(summ, ep)

    def play(self, number_games, env, vae, mcts):
        states=[]
        actions=[]
        rewards=[]
        isTerminal=[]
        for i in range(number_games):
            s,d=env.initializeGame()
            while not(d):
                s=preprocessingState(s)
                #Returns the embedding of the current state
                s=np.squeeze(vae.encode(s))
                a=mcts.predict(s)

                s1, r, d=env.repeatStep(a+1)

                states.append(s)
                actions.append(a)
                rewards.append(r)
                isTerminal.append(d)
                s=s1
        return states, actions, rewards, isTerminal

    #Used to retrieve a sequence of 10 timesteps and action plus the target state embedding
    def prepareRNNData(self, batch_size, timesteps, features, dataset):
        inputData=np.zeros((batch_size, timesteps, features + 1))#+1 is action
        labelData=np.zeros((batch_size, timesteps, features +1))#+1 is reward

        num_prevSteps=(timesteps-1)

        #random select 32 states (-1 to prevent to retrieve the last future action)
        s_idxs=np.random.randint(timesteps, (dataset[0].shape[0]-1), 32)
        for i,s_i in enumerate(s_idxs):
            #retrieve the timesteps-1 previous states and actions
            seqStates=dataset[0][(s_i-num_prevSteps):(s_i+1)]
            seqActions=np.expand_dims(dataset[1][(s_i-num_prevSteps):(s_i+1)], axis=-1)
            inputData[i]=np.concatenate((seqStates, seqActions), axis=-1)
            #retrieve the states and actions shifted in the future by 1
            seqStates=dataset[0][(s_i-num_prevSteps+1):(s_i+2)]
            seqActions=np.expand_dims(dataset[1][(s_i-num_prevSteps+1):(s_i+2)], axis=-1)
            labelData[i]=np.concatenate((seqStates, seqActions), axis=-1)
        '''

        num_prevSteps=(timesteps-1)
        #Store the idx of terminal states. Avoid training
        #using a sequence coming from 2 different games
        terminal_idxs=np.argwhere(np.asarray(dataset[1])==-1)
        #Generate the timesteps for each batch
        i=0
        while (i < batch_size):
            idx=np.random.randint(0, (len(dataset[0])-timesteps-1))

            #check that it is not terminal state
            if(not(np.any(terminal_idxs==idx))):
                if(idx<num_prevSteps):
                    #pad the first transitions with zeros 
                    states=np.asarray(dataset[0][:(idx+1)])
                    num_pads=(timesteps-states.shape[0])
                    zeros_pad=np.zeros((num_pads, features))
                    states=np.concatenate((zeros_pad, states), axis=0)

                    #pad the first actions with zeros
                    actions=np.expand_dims(np.asarray(dataset[1][:(idx+1)]), axis=-1)
                    zeros_pad=np.zeros((num_pads,1))
                    actions=np.concatenate((zeros_pad, actions), axis=0)
                else:
                    #retrieve state and the 9 previous states
                    states=np.asarray(dataset[0][(idx-num_prevSteps):(idx+1)])
                    #retrieve the action and the 9 previous actions
                    actions=np.expand_dims(np.asarray(dataset[1][(idx-num_prevSteps):(idx+1)]), axis=-1)

                inputData[i]=np.concatenate((states, actions), axis=-1)

                #Retrieve the target state s' and the reward coming from s+a
                labelData[i]=np.array(np.append(dataset[0][idx+1], dataset[2][idx]))
                i+=1
        '''
        return inputData, labelData