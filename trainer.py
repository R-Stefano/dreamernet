import tensorflow as tf
import numpy as np
import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

class Trainer():
    def prepareVAEGAN(self,frames, vae):
        VAE_epoches=FLAGS.VAE_training_epoches +1
        GAN_epoches=FLAGS.VAEGAN_epoches +1
        GAN_disc_train_real_epoches=FLAGS.VAEGAN_disc_real_epoches +1
        GAN_disc_train_fake_epoches=FLAGS.VAEGAN_disc_fake_epoches +1
        GAN_gen_train_epoches=FLAGS.VAEGAN_gen_epoches +1

        train_batch_size=FLAGS.VAEGAN_train_size
        test_batch_size=FLAGS.VAEGAN_test_size

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

        if(FLAGS.training_VAEGAN):
            for ep in range(GAN_epoches):
                print('Training GAN, epoch ({}/{})'.format(ep,GAN_epoches))
                #First train the discriminator on real images
                for i in range(GAN_disc_train_real_epoches):
                    global_step=(ep*GAN_epoches) + i
                    #feed images and teach discrimantor that are real (label 0-0.1)
                    #Sample from frames generated
                    idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
                    batchData=train_dataset[idxs]/255.

                    real_labels=np.random.random(size=train_batch_size)*0.1
                    _, summ = vae.sess.run([vae.disc_opt, vae.training_discriminator_real], feed_dict={vae.gen_output: batchData,
                                                                                                    vae.disc_Y: real_labels})
                    vae.file.add_summary(summ, global_step)

                    if i%5==0:
                        idxs=np.random.randint(0, test_dataset.shape[0], size=test_batch_size)
                        batchData=test_dataset[idxs]/255.

                        summ = vae.sess.run(vae.testing_discriminator_real, feed_dict={vae.gen_output: batchData})
                        vae.file.add_summary(summ, global_step)

                #Second train the discriminator on fake images(vae output)
                for i in range(GAN_disc_train_fake_epoches):
                    global_step=(ep*GAN_epoches) + i
                    #feed images and teach discrimantor that are fake (label 0.9-1)
                    #Sample from frames generated
                    idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
                    batchData=train_dataset[idxs]

                    real_labels=np.random.random(size=train_batch_size)*0.1+0.9
                    _,summ = vae.sess.run([vae.disc_opt, vae.training_discriminator_fake], feed_dict={vae.gen_X: batchData,
                                                                                                    vae.disc_Y: real_labels})
                    vae.file.add_summary(summ, global_step)
                    if i%5==0:
                        idxs=np.random.randint(0, test_dataset.shape[0], size=test_batch_size)
                        batchData=test_dataset[idxs]

                        summ = vae.sess.run(vae.testing_discriminator_fake, feed_dict={vae.gen_X: batchData})
                        vae.file.add_summary(summ, global_step)

                #Third train generator to fool discriminator
                for i in range(GAN_gen_train_epoches):
                    global_step=(ep*GAN_epoches) + i
                    #feed images and teach Generator to fool discriminator
                    #Sample from frames generated
                    idxs=np.random.randint(0, train_dataset.shape[0], size=train_batch_size)
                    batchData=train_dataset[idxs]

                    _,summ = vae.sess.run([vae.gen_opt, vae.training_generator], feed_dict={vae.gen_X: batchData})
                    vae.file.add_summary(summ, global_step)

                    if i%5==0:
                        idxs=np.random.randint(0, test_dataset.shape[0], size=test_batch_size)
                        batchData=test_dataset[idxs]

                        summ = vae.sess.run(vae.testing_generator, feed_dict={vae.gen_X: batchData})
                        vae.file.add_summary(summ, global_step)

                vae.save()


    def prepareRNN(self,frames, actions, rewards, rnn, vaegan):
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
            inputData, labelData = utils.prepareRNNData(train_batch_size, rnn.sequence_length, rnn.latent_dimension, train_dataset['embeds'], train_dataset['actions'], train_dataset['rews'], vaegan)
            
            #initialize hidden state and cell state to zeros 
            init_state=np.zeros((rnn.num_layers, 2, train_batch_size, rnn.hidden_units))

            #Train
            
            _, summ = rnn.sess.run([rnn.opt, rnn.training], feed_dict={rnn.X: inputData, 
                                                         rnn.true_next_state: labelData,
                                                         rnn.init_state: init_state})

            rnn.file.add_summary(summ, ep)
            #Saving and testing
            if ep % 50 ==0:
                inputData, labelData = utils.prepareRNNData(test_batch_size, rnn.sequence_length, rnn.latent_dimension, test_dataset['embeds'], test_dataset['actions'], test_dataset['rews'], vaegan)

                #initialize hidden state and cell state to zeros 
                init_state=np.zeros((rnn.num_layers, 2, test_batch_size, rnn.hidden_units))

                #Train
                summ= rnn.sess.run(rnn.testing, feed_dict={rnn.X: inputData, 
                                                           rnn.true_next_state: labelData,
                                                           rnn.init_state: init_state})
                rnn.file.add_summary(summ, ep)

                #DISPLAY NETWORK PROGESSION IN TENSORFLOW
                #take randomly 3 sequences from testing data in order to get the s' prediction
                idxs=np.random.choice(test_batch_size, 3)
                predictingExamples=inputData[idxs]
                out, h=rnn.predict(predictingExamples)
                out=out.reshape(3, rnn.sequence_length, -1)

                #retrieve the last predicted state as well as the last inputdata state
                s1preds=out[:,-1,:-1]
                sinput=predictingExamples[:,-1,:-1]
                
                #make the vaegan decode them
                frames=vaegan.decode(np.concatenate((sinput, s1preds), axis=0))

                #display in tensorboard
                summ= rnn.sess.run(rnn.predicting, feed_dict={rnn.frame_s: frames[:3], 
                                                            rnn.frame_s1: frames[3:]})
                rnn.file.add_summary(summ, ep)
                
                print('Saving RNN..')
                rnn.save()

    def trainSystem(self, statesBuffer, 
                    actionsBuffer, 
                    rewardsBuffer, 
                    terminalBuffer, 
                    hidden_states, 
                    tdErrorBuffer,
                    vaegan, 
                    rnn,
                    actor,
                    step):
        '''
        #1. TRAIN VAE using frames
        idxs=np.random.randint(0, len(statesBuffer), 32)
        states=np.asarray(statesBuffer)[idxs]
        
        _, summ=vae.sess.run([vae.opt, vae.playing], feed_dict={vae.X: states})
        vae.file.add_summary(summ, i)
        vae.save()
        ''' 
        #2. TRAIN RNN using states, action, next state, reward
        inputData, labelData=utils.prepareRNNData(rnn.train_size, 
                                                  rnn.sequence_length, 
                                                  rnn.latent_dimension, 
                                                  statesBuffer,
                                                  actionsBuffer, 
                                                  rewardsBuffer,
                                                  vaegan)
           
        initState=np.zeros((rnn.num_layers, 2, rnn.train_size, rnn.hidden_units))
        
        _, summ=rnn.sess.run([rnn.opt, rnn.playing], feed_dict={rnn.X: inputData,
                                                                rnn.true_next_state: labelData,
                                                                rnn.init_state: initState})
        rnn.file.add_summary(summ, step)
        rnn.save()

        #3. TRAIN ACTOR with s,h and real reward
        if (FLAGS.use_prioritized_exp_rep):
            #Get higher td errors and lower td errors. Use argsort 10k elements is still fast to sort
            higher_erros_idxs=np.asarray(np.argsort(tdErrorBuffer)[-26:])
            low_erros_idxs=np.asarray(np.argsort(tdErrorBuffer)[:6])
            idxs=np.concatenate((low_erros_idxs, higher_erros_idxs), axis=-1)
        else:
            #retrieve transictions to use for training
            idxs=np.random.randint(1, statesBuffer.shape[0]-1, 32)

        #First, feed states for vs1
        states=vaegan.encode(statesBuffer[idxs])
        if (FLAGS.prediction_type == 'KL'):
            mu, std=np.split(states, [rnn.latent_dimension], axis=-1)
            states=mu + std*np.random.normal(size=(states.shape[0], rnn.latent_dimension))
        h_state=hidden_states[idxs][:,0,0,0]

        #retrieve maxQ(s',a)
        policy, _=actor.predict(np.concatenate((states, h_state), axis=-1))
        s1avalue=np.max(policy, axis=-1)

        #retrieve the previous transition
        idxs=idxs-1

        #compute value function
        r=rewardsBuffer[idxs]
        d=terminalBuffer[idxs]

        target=((1-d)*0.99 * s1avalue) + r

        #retrieve actions executed at state s
        input_actions=actionsBuffer[idxs]

        states=vaegan.encode(statesBuffer[idxs])
        if (FLAGS.prediction_type == 'KL'):
            mu, std=np.split(states, [rnn.latent_dimension], axis=-1)
            states=mu + std*np.random.normal(size=(states.shape[0], rnn.latent_dimension))
        h_state=hidden_states[idxs][:,0,0,0]

        #train the network
        dict_input= {actor.X: np.concatenate((states, h_state), axis=-1),
                     actor.targets: target,
                     actor.actions: input_actions}

        _, summ, tdErrors=actor.sess.run([actor.opt, actor.training, actor.tdError], feed_dict=dict_input)
        actor.file.add_summary(summ, step)
        actor.save()

        #Update the td errors in the experience replay
        tdErrorBuffer[idxs+1]=tdErrors

        return tdErrorBuffer.tolist()