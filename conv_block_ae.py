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

import h5py

#init = 'he_normal'
init = 'truncated_normal'

class BlockAE:
    def __init__(self,input_shape,block_size,intermediate_dim,latent_dim,loss_type,num_layers=1,num_classes=10):

        self.input_shape = input_shape
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.blocks_y = self.input_shape[0]/block_size
        self.blocks_x = self.input_shape[1]/block_size
        self.blocks_per_image = self.blocks_y * self.blocks_x

        self.x = Input(shape=input_shape,name='input')
        res = Conv2D(intermediate_dim, block_size, strides=block_size, activation='relu', kernel_initializer=init, name='encoder_0')(self.x)
        for i in range(self.num_layers-1):
            res = Conv2D(intermediate_dim, 1, activation='relu', kernel_initializer=init, name='encoder_%d'%(i+1))(res)
        h = res
        self.z = Conv2D(latent_dim, 1, kernel_initializer=init, name='encoder_mean')(h)

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = []
        for i in range(self.num_layers):
            self.decoder_h.append( Conv2D(intermediate_dim, 1, activation='relu', kernel_initializer=init, name='decoder_%d'%i) )
        res = self.z
        for i in range(self.num_layers):
            res = self.decoder_h[i](res)
        h_decoded = res
        if loss_type == 'binary':
          self.decoder_mean = Conv2DTranspose(input_shape[2], block_size, strides=block_size, activation='sigmoid', kernel_initializer=init, name='decoder_mean')
        elif loss_type == 'sad' or loss_type == 'ssd':
          self.decoder_mean = Conv2DTranspose(input_shape[2], block_size, strides=block_size, kernel_initializer=init, name='decoder_mean')
        else:
            raise ValueException('Unknown loss type: %s'%loss_type)
        self.x_decoded_mean = self.decoder_mean(h_decoded)
    
    def xent_loss(self,labels,x_decoded_mean):
        labels = K.batch_flatten(labels)
        x_decoded_mean = K.batch_flatten(x_decoded_mean)
        if self.loss_type == 'binary':
            xent_loss = self.blocks_per_image * keras.losses.binary_crossentropy(labels,x_decoded_mean)
        elif self.loss_type == 'sad':
            xent_loss = K.sum(K.abs(labels - x_decoded_mean), axis=-1)
        elif self.loss_type == 'ssd':
            xent_loss = K.sum(K.square(labels - x_decoded_mean), axis=-1)
        else:
            raise ValueException('Unknown loss type: %s'%self.loss_type)
        return xent_loss

    def ae_loss(self,labels,x_decoded_mean):
        return self.xent_loss(labels,x_decoded_mean)

    def get_ae_model(self):
        return Model(self.x,self.x_decoded_mean)
    
    def get_encoder_model(self):
        return Model(self.x, self.z)

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
    parser.add_argument("--block_size", help="Size of each block for the AE", type=int, metavar="INT")
    parser.add_argument("--intermediate_dim", help="Intermediate dimension", type=int, metavar="INT")
    parser.add_argument("--latent_dim", help="Latent dimension", type=int, metavar="INT")
    parser.add_argument("--num_samples", help="Number of samples to train from", type=int, metavar="INT")
    parser.add_argument("--num_layers", help="Number of layers for the AE", type=int, metavar="INT")
    parser.add_argument("--categorical", help="Make it categorical", type=int, metavar="INT")
    parser.add_argument("--batch_size", help="Number of images per batch", type=int, metavar="INT")
    parser.add_argument("--epochs", help="Total number of epochs", type=int, metavar="INT")
    parser.add_argument("--ae_loss_type", help="AE loss function", type=str, metavar="[sad, ssd, binary, categorical]")
    parser.add_argument("--dataset", help="Dataset name", type=str, metavar="STRING")
    parser.add_argument("--conditional", help="Conditional", type=int, metavar="INT")
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
    loss_type = cfg.ae_loss_type if not args.ae_loss_type else args.ae_loss_type

    block_ae_weights = cfg.block_ae_weights
    results_dir = cfg.results_dir

    block_ae_outputs_dir = cfg.get_bvae_out_path()

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
    elif dataset == 'lfw':
        f = h5py.File('lfw.hdf5','r')
        x = f['data'][:]
        y = f['label'][:]
        f.close()
        num_images = len(x)
        num_train = int(num_images*8/10)
        shuffled_inds = np.random.permutation(num_images)
        train_inds = shuffled_inds[:num_train]
        test_inds = shuffled_inds[num_train:]
        x_train = x[train_inds]
        y_train = y[train_inds]
        x_test = x[test_inds]
        y_test = y[test_inds]
        num_classes = np.max(y[:])+1
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

    block_ae = BlockAE(input_shape=x_train.shape[1:4],
                       block_size=block_size,
                       intermediate_dim=intermediate_dim,
                       latent_dim=latent_dim,
                       loss_type=loss_type,
                       num_layers=cfg.num_layers)

    ae = block_ae.get_ae_model()
    print(ae.summary())
    ae.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss=block_ae.ae_loss)

    if os.path.exists(block_ae_outputs_dir+block_ae_weights):
        ae.load_weights(block_ae_outputs_dir+block_ae_weights)
    else:
        if os.path.exists('./logs'):
            shutil.rmtree('./logs')
        os.makedirs('./logs')
        tb = keras.callbacks.TensorBoard()
        checkpoint = keras.callbacks.ModelCheckpoint(block_ae_outputs_dir+block_ae_weights,save_weights_only=True,period=10)
        ae.fit(x_train,x_train,
                validation_data=[x_test,x_test],
                batch_size=batch_size,
                shuffle=True,
                epochs=epochs,
                callbacks=[tb,checkpoint])
        ae.save_weights(block_ae_outputs_dir+block_ae_weights)

    # build a model to project inputs on the latent space
    encoder = block_ae.get_encoder_model()

    # build a block generator that can sample from the learned distribution
    generator = block_ae.get_decoder_model()

    # pass training images through the ae
    for i in xrange(num_classes):
        ind = np.argmax(all_labels_train[:,i])
        im = x_train[ind]
        imsave(block_ae_outputs_dir+'train_image_%02d.png'%i,np.squeeze(im))

        encoded = encoder.predict(np.expand_dims(im,axis=0))
        pred_im = generator.predict(encoded)
        pred_im = np.clip(pred_im,0,1) * 255.
        pred_im = pred_im.astype('uint8')

        imsave(block_ae_outputs_dir+'pred_train_image_%02d.png'%i,np.squeeze(pred_im))

    # pass testing images through the ae
    for i in xrange(num_classes):
        ind = np.argmax(all_labels_test[:,i])
        im = x_test[ind]
        imsave(block_ae_outputs_dir+'test_image_%02d.png'%i,np.squeeze(im))

        encoded = encoder.predict(np.expand_dims(im,axis=0))
        pred_im = generator.predict(encoded)
        pred_im = np.clip(pred_im,0,1) * 255.
        pred_im = pred_im.astype('uint8')

        imsave(block_ae_outputs_dir+'pred_test_image_%02d.png'%i,np.squeeze(pred_im))

    # sample a random image
    #for i in xrange(num_classes):
        #z_sample = np.random.normal(size=(len(blocks),block_ae.latent_dim))*0.01
        #labels_sample = np.zeros((len(z_sample),num_classes),dtype=bool)
        #labels_sample[:,0] = True
        #if block_ae.loss_type == 'categorical':
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
            #imsave(block_ae_outputs_dir+'sampled_image_%02d.png'%i,pred_im[:,:,0])
        #else:
            #imsave(block_ae_outputs_dir+'sampled_image_%02d.png'%i,pred_im)


