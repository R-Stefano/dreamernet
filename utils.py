import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

flags = tf.app.flags
FLAGS = flags.FLAGS

def saveImage(matrix,name):
    plt.imsave('debug/'+name+'.png', matrix)

def preprocessingState(state):
    # cropping
    s = state[FLAGS.gap:FLAGS.gap + FLAGS.crop_size,:]

    #resizing
    s = cv2.resize(s, dsize=(FLAGS.img_size, FLAGS.img_size), interpolation=cv2.INTER_CUBIC)
    return s

#Called in preprocessing. Used to see what VAEGAN encoded
def testingVAEGAN(frames, vae):
    idxs=np.random.randint(0, frames.shape[0], 4)
    inputs=frames[idxs]
    
    out=vae.sess.run(vae.gen_output, feed_dict={vae.gen_X:inputs})#vae.decode(vae.encode(inputs))
    out=(out*255).astype(int)
    
    f, axarr = plt.subplots(4,2)
    for i in range(out.shape[0]):
        axarr[i,0].imshow(inputs[i])
        axarr[i,1].imshow(out[i])
    plt.show()

#Used to retrieve a sequence of frames and actions plus the next frames and the rewards
def prepareRNNData(batch_size, timesteps, features, frames, actions, rewards, vaegan):
    inputData=np.zeros((batch_size, timesteps, features + FLAGS.actions_size))#+1 is action
    labelData=np.zeros((batch_size, timesteps, features +1))#+1 is reward

    #random select 32 states (-1 to avoid to predict the frame after the last frame)
    s_idxs=np.random.randint(timesteps, (frames.shape[0]-1), batch_size)
    for i,end in enumerate(s_idxs):
        start=end-timesteps
        all_frames=frames[start:end+1]
        
        #frames must be encoded first
        if (all_frames.shape[-1]==3):
            all_frames=vaegan.encode(all_frames)

        #retrieve the timesteps-1 previous states and actions
        seqStates=all_frames[:-1]
        seqActions=np.expand_dims(actions[start:end], axis=-1)
        inputData[i]=np.concatenate((seqStates, seqActions), axis=-1)

        #retrieve the rewards and the states shifted by 1 in the future
        seqStates=all_frames[1:]
        seqRews=np.expand_dims(rewards[start:end], axis=-1)
        labelData[i]=np.concatenate((seqStates, seqRews), axis=-1)

    return inputData, labelData

#Called in preprocessing. Used to see what RNN predicts
def testingRNN(embeds, actions, rewards, rnn, vae):
    idxs=np.random.randint(rnn.sequence_length, embeds.shape[0], 4)
    errors=[]
    for idx in idxs:
        sequenceEmbeds=embeds[(idx-rnn.sequence_length):idx].reshape((1, rnn.sequence_length, embeds.shape[-1]))
        actionLength=actions[(idx-rnn.sequence_length):idx].reshape((1,-1,1))
        inputData=np.concatenate((sequenceEmbeds, actionLength), axis=-1)
        out, h=rnn.predict(inputData)
        #The last 2 states of the sequence are going to be reconstruct
        inputEmbed=(sequenceEmbeds[0,-2:]*255).astype(int)
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
    
#Used to create the sprite to use to visualize the embeddings
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    #number of images per row
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    #total dimension sprite
    spriteimage = np.zeros((img_h * n_plots ,img_w * n_plots,3), dtype=int)
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                #retrieve the image one at the time
                this_img = images[this_filter]
                #copy the image into the sprite
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    return spriteimage

def visualizeEmbeddings(embeds, imgs_to_embed, model_folder):
    #Create embeddings
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    embed_var =tf.Variable(embeds, name="embeddings")
    embedding.tensor_name = "embeddings"

    '''
    #Create the labels for the datapoints
    metadata_filename='labels_embeddings.tsv'
    with open(('models/VAE/'+metadata_filename), 'w') as f:
        labels = ['test' for i in range(32)]
        for c in labels:
            f.write('{}\n'.format(c))
    embedding.metadata_path = 
    '''

    #Create the sprite 
    embedding.sprite.image_path = "embeddingsSprites.png"
    embedding.sprite.single_image_dim.extend([64,64])
    sprite_image = create_sprite_image(imgs_to_embed)
    plt.imsave((model_folder+"embeddingsSprites.png"),sprite_image)

    projector.visualize_embeddings(file,config)

    sess.run(embed_var.initializer)
    embd_saver.save(sess, 'models/VAE/embeddings.ckpt')
    embd_saver=tf.train.Saver([embed_var])