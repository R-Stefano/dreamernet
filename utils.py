import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.contrib.tensorboard.plugins import projector


def saveImage(matrix,name):
    plt.imsave('debug/'+name+'.png', matrix)

def preprocessingState(state):
    # cropping
    s = state[30:-20]

    #resizing
    s = cv2.resize(s, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    return s

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