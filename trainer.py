from random import shuffle
import tensorflow as tf
import numpy as np

class Trainer():
    def trainVAE(self,sess, frames, vae):
        print('Starting VAE training..')
        model_folder='models/VAE/'
        training_epoches=5
        train_batch_size=32
        test_batch_size=64

        #Create file
        file=tf.summary.FileWriter(model_folder, sess.graph)
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        shuffle(frames)

        train_dataset=frames[:int(len(frames)*0.75)]
        test_dataset=frames[int(len(frames)*0.75):]

        for ep in range(len(train_dataset)//train_batch_size):
            batchEnd = (ep+1)*train_batch_size
            batchData=np.asarray(train_dataset[ep*train_batch_size:batchEnd])

            _, summ = sess.run([vae.opt, vae.training], feed_dict={vae.X: batchData/255.})

            file.add_summary(summ, ep)

            if ep % 50 ==0:
                saver.save(sess, model_folder+"graph.ckpt")

                rec_loss=0
                kl_loss=0
                tot_loss=0
                avg =0
                for batchStart in range(0, len(test_dataset), test_batch_size):
                    batchEnd=batchStart+test_batch_size
                    batchData=np.asarray(test_dataset[batchStart:batchEnd])

                    l1, l2, l3= sess.run([vae.reconstr_loss, vae.KLLoss, vae.totLoss], feed_dict={vae.X: batchData/255.})

                    rec_loss +=l1
                    kl_loss += l2
                    tot_loss += l3
                    avg +=1

                summ = sess.run(vae.testTensorboard, feed_dict={vae.recLossPlace: (rec_loss/avg), vae.klLossPlace: (kl_loss/avg), vae.totLossPlace: (tot_loss/avg)})
                file.add_summary(summ, ep)
