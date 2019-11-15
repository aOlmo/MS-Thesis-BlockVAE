#!/usr/bin/env python

import cv2
import os
import shutil
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.stats import norm
from tqdm import trange

import keras
from keras.layers import Input, Dense, Lambda, Layer, Activation, Reshape, Add, Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import tensorflow as tf

import argparse
from configs import Configs
from skimage.measure import compare_ssim as ssim

class BlockVAE:
    def __init__(self,input_shape,block_size,intermediate_dim,latent_dim,num_layers=1,num_classes=10):

        self.input_shape = input_shape
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.blocks_y = self.input_shape[0]/block_size
        self.blocks_x = self.input_shape[1]/block_size
        self.blocks_per_image = self.blocks_y * self.blocks_x

        self.x = Input(shape=input_shape,name='input')
        res = Conv2D(intermediate_dim, block_size, strides=block_size, activation='relu', kernel_initializer='he_normal', name='encoder_0')(self.x)
        for i in range(self.num_layers-1):
            res = Conv2D(intermediate_dim, 1, activation='relu', kernel_initializer='he_normal', name='encoder_%d'%(i+1))(res)
        h = res
        self.z_mean = Conv2D(latent_dim, 1, kernel_initializer='he_normal', name='encoder_mean')(h)
        self.z_log_var = Conv2D(latent_dim, 1, kernel_initializer='he_normal', name='encoder_log_var')(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.blocks_y, self.blocks_x, latent_dim), mean=0., stddev=1.0)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling,name='sampler')([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = []
        for i in range(self.num_layers):
            self.decoder_h.append( Conv2D(intermediate_dim, 1, activation='relu', kernel_initializer='he_normal', name='decoder_%d'%i) )
        res = z
        for i in range(self.num_layers):
            res = self.decoder_h[i](res)
        h_decoded = res
        self.decoder_mean = Conv2DTranspose(input_shape[2], block_size, strides=block_size, activation='sigmoid', kernel_initializer='he_normal', name='decoder_mean')
        self.x_decoded_mean = self.decoder_mean(h_decoded)
    
    def xent_loss(self,labels,x_decoded_mean):
        xent_loss = K.binary_crossentropy(labels,x_decoded_mean)
        xent_loss = K.sum(xent_loss,axis=-1)
        xent_loss = K.sum(xent_loss,axis=-1)
        xent_loss = K.sum(xent_loss,axis=-1)
        #xent_loss = xent_loss / self.blocks_per_image
        return xent_loss

    def kl_loss(self,labels,x_decoded_mean):
        kl_loss = -0.5 * (1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var))
        kl_loss = K.sum(kl_loss,axis=-1)
        kl_loss = K.sum(kl_loss,axis=-1)
        kl_loss = K.sum(kl_loss,axis=-1)
        #kl_loss = kl_loss / self.blocks_per_image
        return kl_loss
    
    def vae_loss(self,labels,x_decoded_mean):
        return self.xent_loss(labels,x_decoded_mean) + self.kl_loss(labels,x_decoded_mean)

    def get_vae_model(self):
        return Model(self.x,self.x_decoded_mean)
    
    def get_encoder_model(self):
        return Model(self.x, [self.z_mean,self.z_log_var])

    def get_decoder_model(self):
        decoder_input = Input(shape=(self.blocks_y,self.blocks_x,self.latent_dim))
        res = decoder_input
        for i in range(self.num_layers):
            res = self.decoder_h[i](res)
        _h_decoded = res
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        return Model(decoder_input, _x_decoded_mean)

if __name__ == '__main__':
    from keras import metrics
    from keras.datasets import mnist, cifar10

    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )

    defaults = {}

    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    cfg = Configs(args.conf_file) if args.conf_file else Configs()

    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**defaults)
    parser.add_argument("--nb_epoch", help="Number of epochs", type=int, metavar="INT")
    parser.add_argument("--red_only", help="Use red only", type=int, metavar="INT")
    parser.add_argument("--block_size", help="Size of each block for the VAE", type=int, metavar="INT")
    parser.add_argument("--intermediate_dim", help="Intermediate dimension", type=int, metavar="INT")
    parser.add_argument("--latent_dim", help="Latent dimension", type=int, metavar="INT")
    parser.add_argument("--num_samples", help="Number of samples to train from", type=int, metavar="INT")
    parser.add_argument("--num_layers", help="Number of layers for the VAE", type=int, metavar="INT")
    parser.add_argument("--categorical", help="Make it categorical", type=int, metavar="INT")
    parser.add_argument("--batch_size", help="Number of images per batch", type=int, metavar="INT")
    parser.add_argument("--epochs", help="Total number of epochs", type=int, metavar="INT")
    parser.add_argument("--vae_loss_type", help="VAE loss function", type=str, metavar="[sad, ssd, binary, categorical]")
    parser.add_argument("--dataset", help="Dataset name", type=str, metavar="STRING")
    parser.add_argument("--conditional", help="Conditional", type=int, metavar="INT")
    parser.add_argument("--epsilon_std", help="Epsilon value", type=float, metavar="FLOAT")
    parser.add_argument("--calc_ssim", help="Calcualte average SSIM", type=int, metavar="INT")
    parser.add_argument("--calc_nll", help="Calcualte average NLL ", type=int, metavar="INT")

    args = parser.parse_args(remaining_argv)

    red_only = cfg.red_only
    block_size = cfg.block_size

    if red_only:
        original_dim = block_size*block_size
    else:
        original_dim = block_size*block_size*3

    if original_dim <> cfg.original_dim:
        raise ValueException('Calculated original_dim (%d) does not match configuration file (%d)'%(original_dim,cfg.original_dim))

    #TODO: Each of the following statements can fail when args.<param_name> = 0!
    intermediate_dim = cfg.intermediate_dim if not args.intermediate_dim else args.intermediate_dim
    latent_dim = cfg.latent_dim if not args.latent_dim else args.latent_dim
    batch_size = cfg.batch_size if not args.batch_size else args.batch_size
    epochs = cfg.epochs if not args.epochs else args.epochs
    dataset = cfg.dataset if not args.dataset else args.dataset

    block_vae_weights = cfg.block_vae_weights
    results_dir = cfg.results_dir

    block_vae_outputs_dir = cfg.get_bvae_out_path()

    # load dataset
    if dataset == 'cifar10':
        print('loading cifar10...')
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # add dimension for channels
        x_train = np.expand_dims(x_train,axis=-1)
        x_test = np.expand_dims(x_test,axis=-1)
        num_classes = 10
    else:
        raise ValueException('unknown dataset %s'%dataset)
    
    all_labels_train = keras.utils.to_categorical(y_train)
    all_labels_test = keras.utils.to_categorical(y_test)

    if red_only:
        # select only red channel
        x_train = x_train[:,:,:,[0]]
        x_test = x_test[:,:,:,[0]]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    block_vae = BlockVAE(input_shape=x_train.shape[1:4],block_size=block_size,intermediate_dim=intermediate_dim,latent_dim=latent_dim,num_layers=cfg.num_layers)

    vae = block_vae.get_vae_model()
    print(vae.summary())
    vae.compile(optimizer=keras.optimizers.Adam(0.0001), loss=block_vae.vae_loss, metrics=[block_vae.xent_loss,block_vae.kl_loss])

    if os.path.exists(block_vae_outputs_dir+block_vae_weights):
        vae.load_weights(block_vae_outputs_dir+block_vae_weights)
    else:
        if os.path.exists('./logs'):
            shutil.rmtree('./logs')
        os.makedirs('./logs')
        tb = keras.callbacks.TensorBoard()
        vae.fit(x_train,x_train,
                validation_data=[x_test,x_test],
                batch_size=batch_size,
                shuffle=True,
                epochs=epochs,
                callbacks=[tb])
        vae.save_weights(block_vae_outputs_dir+block_vae_weights)

    # build a model to project inputs on the latent space
    encoder = block_vae.get_encoder_model()

    # build a block generator that can sample from the learned distribution
    generator = block_vae.get_decoder_model()

    # pass training images through the vae
    for i in xrange(num_classes):
        ind = np.argmax(all_labels_train[:,i])
        im = x_train[ind]
        imsave(block_vae_outputs_dir+'train_image_%02d.png'%i,np.squeeze(im))

        encoded, log_var = encoder.predict(np.expand_dims(im,axis=0))
        pred_im = generator.predict(encoded)

        imsave(block_vae_outputs_dir+'pred_train_image_%02d.png'%i,np.squeeze(pred_im))

    # pass testing images through the vae
    for i in xrange(num_classes):
        ind = np.argmax(all_labels_test[:,i])
        im = x_test[ind]
        imsave(block_vae_outputs_dir+'test_image_%02d.png'%i,np.squeeze(im))

        encoded, log_var = encoder.predict(np.expand_dims(im,axis=0))
        pred_im = generator.predict(encoded)

        imsave(block_vae_outputs_dir+'pred_test_image_%02d.png'%i,np.squeeze(pred_im))

    # sample a random image
    #for i in xrange(num_classes):
        #z_sample = np.random.normal(size=(len(blocks),block_vae.latent_dim))*0.01
        #labels_sample = np.zeros((len(z_sample),num_classes),dtype=bool)
        #labels_sample[:,0] = True
        #if block_vae.loss_type == 'categorical':
            #if conditional:
                #probs = generator.predict([z_sample,labels_sample])
            #else:
                #probs = generator.predict(z_sample)
            #blocks = probs_to_blocks(probs)
        #else:
            #if conditional:
                #blocks = generator.predict([z_sample,labels_sample]) + training_mean
            #else:
                #blocks = generator.predict(z_sample) + training_mean
        #pred_im = blocks_to_image(blocks,im.shape[0],im.shape[1],im.shape[2],block_size)
        #if red_only:
            #imsave(block_vae_outputs_dir+'sampled_image_%02d.png'%i,pred_im[:,:,0])
        #else:
            #imsave(block_vae_outputs_dir+'sampled_image_%02d.png'%i,pred_im)


