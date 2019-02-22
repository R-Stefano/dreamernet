from models import RNN, VAEGAN
from trainer import Trainer
from EnvWrapper import EnvWrap

import tensorflow as tf 
import numpy as np
import os
import utils
import shutil #clean folder for retraining

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env, vae, trainer, rnn):
    if(FLAGS.training_VAE and (len(os.listdir('models/VAEGAN/'))!=0) and FLAGS.training_VAEGAN):
        print('cleaning VAE folder..')
        shutil.rmtree('models/VAEGAN/')#clean folder
    if(FLAGS.training_RNN and (len(os.listdir('models/RNN/'))!=0)):
        print('cleaning RNN folder..')
        shutil.rmtree('models/RNN/')#clean folder

    print('Collecting transitions for training..')
    frames, actions, rewards=env.run(FLAGS.games)

    if (FLAGS.training_VAEGAN or FLAGS.training_VAE):
        print('Training VAEGAN')
        trainer.prepareVAEGAN(frames.copy(), vae)


    if (FLAGS.testing_VAEGAN):
        print('Testing VAEGAN')
        utils.testingVAEGAN(frames, vae)

    if (FLAGS.training_RNN):
        print('Training RNN')
        embeds=vae.encode(frames)
        trainer.prepareRNN(embeds, actions, rewards, rnn, vae)

    if(FLAGS.testing_RNN):
        print('Testing RNN')
        embeds=vae.encode(frames)
        utils.testingRNN(embeds, actions, rewards, rnn, vae)