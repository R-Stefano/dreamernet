from random import shuffle
import tensorflow as tf
import numpy as np
from utils import preprocessingState

class Trainer():
    def trainVAE(self,sess, frames, vae):
        print('Starting VAE training..')
        training_epoches=5
        train_batch_size=32
        test_batch_size=64

        #Shuffle the examples
        shuffle(frames)

        #TODO: CONVERT TO NUMPY USING IDXS MASK
        train_dataset=frames[:int(len(frames)*0.75)]
        test_dataset=frames[int(len(frames)*0.75):]

        for ep in range(training_epoches):
            print('Training VAE, epoch ({}/{})'.format(ep,training_epoches))
            #randomly 
            idx=np.random.randint(0, len(train_dataset)-train_batch_size)
            batchData=np.asarray(train_dataset[idx: idx+train_batch_size])

            _, summ = sess.run([vae.opt, vae.training], feed_dict={vae.X: batchData})

            vae.file.add_summary(summ, ep)

            if ep % 50 ==0:
                print('Saving VAE..')
                vae.saver.save(sess, vae.model_folder+"graph.ckpt")

                rec_loss=0
                kl_loss=0
                tot_loss=0
                avg =0
                print('Testing VAE..')
                for batchStart in range(0, len(test_dataset), test_batch_size):
                    batchEnd=batchStart+test_batch_size
                    batchData=np.asarray(test_dataset[batchStart:batchEnd])

                    l1, l2, l3= sess.run([vae.reconstr_loss, vae.KLLoss, vae.totLoss], feed_dict={vae.X: batchData})

                    rec_loss +=l1
                    kl_loss += l2
                    tot_loss += l3
                    avg +=1

                summ = sess.run(vae.testTensorboard, feed_dict={vae.recLossPlace: (rec_loss/avg), vae.klLossPlace: (kl_loss/avg), vae.totLossPlace: (tot_loss/avg)})
                vae.file.add_summary(summ, ep)

    def trainRNN(self,sess, frames, actions, rnn):
        print('Starting RNN training..')
        training_epoches=5
        train_batch_size=32
        test_batch_size=64

        train_dataset=(frames[:int(len(frames)*0.75)], actions[:int(len(frames)*0.75)])
        test_dataset=(frames[int(len(frames)*0.75):], actions[int(len(frames)*0.75):])

        for ep in range(training_epoches):
            print('Training RNN, epoch ({}/{})'.format(ep,training_epoches))
            inputData, labelData = self.prepareRNNData(train_batch_size, rnn.timesteps, rnn.state_rep_length, train_dataset)
            
            #initialize hidden state and cell state to zeros 
            cell_s=np.zeros((train_batch_size, rnn.hidden_units))
            hidden_s=np.zeros((train_batch_size, rnn.hidden_units))

            #Train
            _, summ = sess.run([rnn.opt, rnn.training], feed_dict={rnn.X: inputData, 
                                                         rnn.true_next_state: labelData,
                                                         rnn.cell_state: cell_s,
                                                         rnn.hidden_state: hidden_s})
            rnn.file.add_summary(summ, ep)

            #Saving and testing
            if ep % 50 ==0:
                print('Saving RNN..')
                rnn.saver.save(sess, rnn.model_folder+"graph.ckpt")

                totLoss=0
                avg=0
                print('Testing RNN..')
                for b in range(len(test_dataset[0])//test_batch_size):
                    inputData, labelData = self.prepareRNNData(test_batch_size, rnn.timesteps, rnn.state_rep_length, test_dataset)

                    #initialize hidden state and cell state to zeros 
                    cell_s=np.zeros((test_batch_size, rnn.hidden_units))
                    hidden_s=np.zeros((test_batch_size, rnn.hidden_units))

                    #Train
                    loss = sess.run([rnn.loss], feed_dict={rnn.X: inputData, 
                                                           rnn.true_next_state: labelData,
                                                           rnn.cell_state: cell_s,
                                                           rnn.hidden_state: hidden_s})
                    totLoss += loss[0]
                    avg +=1

                summ = sess.run(rnn.testing, feed_dict={rnn.totLossPlace: (totLoss/avg)})
                rnn.file.add_summary(summ, ep)

    #Called to train alphazero using everything.
    def trainAlphaZero(self, mcts, vae, rnn, env):
        training_games=1

        for i in range(training_games):
            s=preprocessingState(env.initializeGame())
            #Returns the embedding of the current state
            s=np.squeeze(vae.predict(s))
            a=mcts.predict(s)

    
    #Used to retrieve a sequence of 10 timesteps and action plus the target state embedding
    def prepareRNNData(self, batch_size, timesteps, features, dataset):
        inputData=np.zeros((batch_size, timesteps, features + 1))
        labelData=np.zeros((batch_size, features))

        #Store the idx of terminal states. Avoid training
        #using a sequence ocming from 2 different games
        terminal_idxs=np.argwhere(np.asarray(dataset[1])==-1)

        #Generate the timesteps for each batch
        i=0
        while (i < batch_size):
            idx=np.random.randint(0, (len(dataset[0])-timesteps-1))

            #check that it is not terminal state
            if(not(np.any(terminal_idxs==idx))):
                if(idx<9):
                    #pad the first transitions with zeros 
                    states=np.asarray(dataset[0][:(idx+1)])
                    num_pads=(10-states.shape[0])
                    zeros_pad=np.zeros((num_pads, features))
                    states=np.concatenate((zeros_pad, states), axis=0)

                    #pad the first actions with zeros
                    actions=np.expand_dims(np.asarray(dataset[1][:(idx+1)]), axis=-1)
                    zeros_pad=np.zeros((num_pads,1))
                    actions=np.concatenate((zeros_pad, actions), axis=0)
                else:
                    #retrieve state and the 9 previous states
                    states=np.asarray(dataset[0][(idx-9):(idx+1)])
                    #retrieve the action and the 9 previous actions
                    actions=np.expand_dims(np.asarray(dataset[1][(idx-9):(idx+1)]), axis=-1)

                inputData[i]=np.concatenate((states, actions), axis=-1)

                #Retrieve the target state
                labelData[i]=np.array(dataset[0][idx+1])
                i+=1
        
        return inputData, labelData