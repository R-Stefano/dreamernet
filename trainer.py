import tensorflow as tf
import numpy as np
from utils import preprocessingState

flags = tf.app.flags
FLAGS = flags.FLAGS

class Trainer():
    def prepareVAEGAN(self,frames, vae):
        VAE_epoches=FLAGS.VAE_training_epoches +1
        GAN_epoches=FLAGS.GAN_epoches +1
        GAN_disc_train_real_epoches=FLAGS.GAN_disc_real_epoches +1
        GAN_disc_train_fake_epoches=FLAGS.GAN_disc_fake_epoches +1

        train_batch_size=FLAGS.VAE_train_size
        test_batch_size=FLAGS.VAE_test_size

        np.random.shuffle(frames)

        train_dataset=frames[:int(len(frames)*0.75)]
        test_dataset=frames[int(len(frames)*0.75):]

        if(FLAGS.training_VAE):
            print('Starting VAE training..')
            #First train the VAE to get a good reconstruction
            for ep in range(VAE_epoches):
                print('Training VAE, epoch ({}/{})'.format(ep,VAE_epoches))

                #Sample from frames generated
                idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
                batchData=train_dataset[idxs]

                _, summ = vae.sess.run([vae.vae_opt, vae.training_vae], feed_dict={vae.gen_X: batchData})

                vae.file.add_summary(summ, ep)

                if ep % 50 ==0:                   
                    print('Testing VAE..')
                    idxs=np.random.randint(0, test_dataset.shape[0], size=test_batch_size)
                    batchData=test_dataset[idxs]

                    summ= vae.sess.run(vae.testing_vae, feed_dict={vae.gen_X: batchData})

                    vae.file.add_summary(summ, ep)

                    vae.save()

        if(FLAGS.training_GAN):
            print('Starting GAN training..')
            for ep in range(GAN_epoches):
                #First train the discriminator on real images
                for i in range(GAN_disc_train_real_epoches):
                    global_step=(ep*GAN_epoches) + i
                    print('Training GAN discriminator on real data, epoch ({}/{})'.format(i,GAN_disc_train_real_epoches))
                    #feed images and teach discrimantor that are real (label 0-0.1)
                    #Sample from frames generated
                    idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
                    batchData=train_dataset[idxs]/255.

                    real_labels=np.random.random(size=train_batch_size)*0.1
                    _, summ = vae.sess.run([vae.disc_opt, vae.training_discriminator_real], feed_dict={vae.gen_output: batchData,
                                                                                                    vae.disc_Y: real_labels})
                    vae.file.add_summary(summ, global_step)

                    if i%5==0:
                        print('Testing Discriminator on real data..')
                        idxs=np.random.randint(0, test_dataset.shape[0], size=test_batch_size)
                        batchData=test_dataset[idxs]/255.

                        _,summ = vae.sess.run([vae.real_acc,vae.testing_discriminator_real], feed_dict={vae.gen_output: batchData})
                        vae.file.add_summary(summ, global_step)

                        vae.save()

                #Second train the discriminator on fake images(vae output)
                for i in range(GAN_disc_train_fake_epoches):
                    global_step=(ep*GAN_epoches) + i
                    print('Training GAN discriminator on fake data, epoch ({}/{})'.format(i,GAN_disc_train_fake_epoches))
                    #feed images and teach discrimantor that are fake (label 0.9-1)
                    #Sample from frames generated
                    idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
                    batchData=train_dataset[idxs]

                    real_labels=np.random.random(size=train_batch_size)*0.1+0.9
                    _, summ = vae.sess.run([vae.gan_opt, vae.training_discriminator_fake], feed_dict={vae.gen_X: batchData,
                                                                                                    vae.disc_Y: real_labels})
                    vae.file.add_summary(summ, global_step)

                    if i%5==0:
                        print('Testing Discriminator on fake data..')
                        idxs=np.random.randint(0, test_dataset.shape[0], size=test_batch_size)
                        batchData=test_dataset[idxs]

                        _,summ = vae.sess.run([vae.fake_acc,vae.testing_discriminator_fake], feed_dict={vae.gen_X: batchData})
                        vae.file.add_summary(summ, global_step)

                        vae.save()


    def prepareRNN(self,frames, actions, rewards, rnn):
        print('Starting RNN training..')
        training_epoches=FLAGS.RNN_training_epoches +1
        train_batch_size=FLAGS.RNN_train_size
        test_batch_size=FLAGS.RNN_test_size

        train_dataset={'embeds':frames[:int(len(frames)*0.75)], 
                       'actions': actions[:int(len(frames)*0.75)], 
                       'rews': rewards[:int(len(frames)*0.75)]}

        test_dataset={'embeds':frames[int(len(frames)*0.75):],
                      'actions':actions[int(len(frames)*0.75):], 
                      'rews': rewards[int(len(frames)*0.75):]}

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
                for b in range(test_dataset['embeds'].shape[0]//test_batch_size):
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

    #Used to retrieve a sequence of 10 timesteps and action plus the target state embedding
    def prepareRNNData(self, batch_size, timesteps, features, dataset):
        inputData=np.zeros((batch_size, timesteps, features + FLAGS.actions_size))#+1 is action
        labelData=np.zeros((batch_size, timesteps, features +1))#+1 is reward

        #random select 32 states (-1 to avoid to predict the frame after the last frame)
        s_idxs=np.random.randint(timesteps, (dataset['embeds'].shape[0]-1), 32)

        for i,end in enumerate(s_idxs):
            start=end-timesteps
            #retrieve the timesteps-1 previous states and actions
            seqStates=dataset['embeds'][start:end]
            seqActions=np.expand_dims(dataset['actions'][start:end], axis=-1)
            inputData[i]=np.concatenate((seqStates, seqActions), axis=-1)

            #retrieve the rewards and the states shifted by 1 in the future
            seqStates=dataset['embeds'][start+1:end+1]
            seqRews=np.expand_dims(dataset['rews'][start:end], axis=-1)
            labelData[i]=np.concatenate((seqStates, seqRews), axis=-1)
        return inputData, labelData


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