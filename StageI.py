import os
import pickle
import random
import time

from progressbar import *
import numpy as np
import tensorflow as tf

from keras import Input, Model
from keras import backend as K
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation, \
    concatenate, Flatten, Lambda, Concatenate
from keras.optimizers import Adam
import model_df

"""# Main File"""

if __name__ == '__main__':

    image_size = 64
    batch_size = 32
    z_dim = 100
    stage1_generator_lr = 0.002
    stage1_discriminator_lr = 0.002
    stage1_lr_decay_step = 400
    epochs = 500
    condition_dim = 128

    # Define optimizers
    dis_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
    gen_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)

    """"
    Load datasets
    """

    with open('image_data.pickle', 'rb') as r:
        image_data = pickle.load(r)

    with open('caption_data.pickle', 'rb') as c:
        caption_data = pickle.load(c)

    print("image_data:{}".format(image_data.shape))
    print("caption_data:{}".format(caption_data.shape))

    """
    Build and compile networks
    """

    input_layer, out = model_df.build_embedding()
    #ca_model = model_df.build_ca_model(input_layer, out)
    #ca_model.compile(loss="binary_crossentropy", optimizer="adam")

    stage1_dis = model_df.build_stage1_discriminator(input_layer, out)
    stage1_dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)
    stage1_dis.summary()

    stage1_gen = model_df.build_stage1_generator(input_layer, out)
    stage1_gen.compile(loss="mse", optimizer=gen_optimizer)

    #embedding_compressor_model = model_df.build_embedding_compressor_model(input_layer, out)
    #embedding_compressor_model.compile(loss="binary_crossentropy", optimizer="adam")

    adversarial_model = model_df.build_adversarial_model(gen_model=stage1_gen, dis_model=stage1_dis)
    adversarial_model.compile(loss=['binary_crossentropy', model_df.KL_loss], loss_weights=[1, 2.0],
                              optimizer=gen_optimizer, metrics=None)


    # Generate an array containing real and fake values
    # Apply label smoothing as well
    real_labels = np.ones((batch_size, 1), dtype=float) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=float) * 0.1

    number_of_batches = int(image_data.shape[0] / batch_size)

    test_image_data = image_data[(number_of_batches - 1) * batch_size: number_of_batches * batch_size]
    test_caption_data = caption_data[(number_of_batches - 1) * batch_size: number_of_batches * batch_size]
    gen_losses = []
    dis_losses = []
    test_gen_losses = []
    test_dis_losses = []

    for epoch in range(epochs):
        print('epoch: ', epoch)
        # Load data and train model
        widgets = ['epoch: '+str(epoch), ' ', Bar(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets)
        for index in progress(range(number_of_batches - 1)):
            print("Batch:{}".format(index + 1))

            """
            Train the discriminator network
            """
            # Sample a batch of data
            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            image_batch = image_data[index * batch_size:(index + 1) * batch_size]
            embedding_batch = caption_data[index * batch_size:(index + 1) * batch_size]
            image_batch = (image_batch - 127.5) / 127.5

            # Generate fake images
            fake_images, _ = stage1_gen.predict([embedding_batch, z_noise], verbose=3)

            # Generate compressed embeddings
            # compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            # compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            # compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

            dis_loss_real = stage1_dis.train_on_batch([image_batch, embedding_batch],
                                                      np.reshape(real_labels, (batch_size, 1)))
            dis_loss_fake = stage1_dis.train_on_batch([fake_images, embedding_batch],
                                                      np.reshape(fake_labels, (batch_size, 1)))
            dis_loss_wrong = stage1_dis.train_on_batch([image_batch[:(batch_size - 1)], embedding_batch[1:]],
                                                       np.reshape(fake_labels[1:], (batch_size - 1, 1)))

            d_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))

            #print("d_loss_real:{}".format(dis_loss_real))
            #print("d_loss_fake:{}".format(dis_loss_fake))
            #print("d_loss_wrong:{}".format(dis_loss_wrong))
            #print("d_loss:{}".format(d_loss))

            """
            Train the generator network 
            """
            g_loss = adversarial_model.train_on_batch([embedding_batch, z_noise, embedding_batch],
                                                      [np.ones((batch_size, 1)) * 0.9, np.ones((batch_size, 256)) * 0.9])
            #print("g_loss:{}".format(g_loss))

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

            # Sample a batch of data
            test_z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            test_image_batch = test_image_data
            test_embedding_batch = test_caption_data
            test_image_batch = (test_image_batch - 127.5) / 127.5

            # Generate fake images
            test_fake_images, _ = stage1_gen.predict([test_embedding_batch, test_z_noise], verbose=3)

            # Generate compressed embeddings
            #test_compressed_embedding = embedding_compressor_model.predict_on_batch(test_embedding_batch)
            #test_compressed_embedding = np.reshape(test_compressed_embedding, (-1, 1, 1, condition_dim))
            #test_compressed_embedding = np.tile(test_compressed_embedding, (1, 4, 4, 1))

            test_dis_loss_real = stage1_dis.evaluate([test_image_batch, test_embedding_batch],
                                                     np.reshape(real_labels, (batch_size, 1)), verbose=0)
            test_dis_loss_fake = stage1_dis.evaluate([test_fake_images, test_embedding_batch],
                                                     np.reshape(fake_labels, (batch_size, 1)), verbose=0)
            test_dis_loss_wrong = stage1_dis.evaluate(
                [test_image_batch[:(batch_size - 1)], test_embedding_batch[1:]],
                np.reshape(fake_labels[1:], (batch_size - 1, 1)), verbose=0)

            test_d_loss = 0.5 * np.add(test_dis_loss_real, 0.5 * np.add(test_dis_loss_wrong, test_dis_loss_fake))

            # print("d_loss_real:{}".format(dis_loss_real))
            # print("d_loss_fake:{}".format(dis_loss_fake))
            # print("d_loss_wrong:{}".format(dis_loss_wrong))
            # print("d_loss:{}".format(d_loss))

            """
            Train the generator network 
            """
            test_g_loss = adversarial_model.evaluate([test_embedding_batch, test_z_noise, test_embedding_batch],
                                                     [np.ones((batch_size, 1)) * 0.9, np.ones((batch_size, 256)) * 0.9], verbose=0)
            # print("g_loss:{}".format(g_loss))

            test_dis_losses.append(test_d_loss)
            test_gen_losses.append(test_g_loss)

        if epoch % 100 == 0 and epoch != 0:
            os.mkdir("./result-" + str(epoch))
            stage1_gen.save("./result-" + str(epoch) + "/stage1_gen-" + str(epoch) + ".h5")
            stage1_dis.save("./result-" + str(epoch) + "/stage1_dis-" + str(epoch) + ".h5")
            with open("./result-" + str(epoch) + "/dis_loss-" + str(epoch) + ".pickle", "wb") as dis:
                pickle.dump(dis_losses, dis)
            with open("./result-" + str(epoch) + "/gen_loss-" + str(epoch) + ".pickle", "wb") as gen:
                pickle.dump(gen_losses, gen)
            with open("./result-" + str(epoch) + "/test_dis_loss-" + str(epoch) + ".pickle", "wb") as test_dis:
                pickle.dump(test_dis_losses, test_dis)
            with open("./result-" + str(epoch) + "/test_gen_loss-" + str(epoch) + ".pickle", "wb") as test_gen:
                pickle.dump(test_gen_losses, test_gen)
