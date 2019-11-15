#!/usr/bin/env python
import numpy as np
np.random.seed(0)

import cv2
import os
import sys
import shutil
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.stats import norm
from tqdm import trange

import keras
from keras.layers import Input, Dense, Lambda, Layer, Activation, Reshape, Add
from keras.models import Model
from keras import backend as K
import tensorflow as tf

import argparse
from configs import Configs
from skimage.measure import compare_ssim as ssim
import time

import h5py

#init = 'he_normal'
init = 'truncated_normal'

class BlockVAE:
    def __init__(self,original_dim,intermediate_dim,latent_dim,num_layers, loss_type,epsilon_std,conditional=False,num_classes=10):

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.loss_type = loss_type
        self.conditional = conditional
        self.num_classes = num_classes

        self.x = Input(shape=(original_dim,),name='input')
        if conditional:
            self.h = Input(shape=(num_classes,),name='label')
        res = self.x
        for i in range(self.num_layers):
            res = Dense(intermediate_dim, activation='relu', kernel_initializer=init, name='encoder_%d'%i)(res)
            if conditional:
                b = Dense(intermediate_dim, name='encoder_bias_%d'%i)(self.h)
                res = Add(name='encoder_bias_add_%d'%i)([res,b])
        self.z_mean = Dense(latent_dim, kernel_initializer=init, name='encoder_mean')(res)
        self.z_log_var = Dense(latent_dim, kernel_initializer=init, name='encoder_log_var')(res)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                      stddev=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling,name='sampler')([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = []
        if conditional:
            self.decoder_b = []
        for i in range(self.num_layers):
            self.decoder_h.append( Dense(intermediate_dim, activation='relu', kernel_initializer=init, name='decoder_%d'%i) )
            if conditional:
                self.decoder_b.append( Dense(intermediate_dim, name='decoder_bias_%d'%i) )
        res = z
        for i in range(self.num_layers):
            res = self.decoder_h[i](res)
            if conditional:
                res = Add(name='decoder_bias_add_%d'%i)([res,self.decoder_b[i](self.h)])
        h_decoded = res
        if loss_type == 'categorical':
            self.decoder_mean = Dense(original_dim*256, kernel_initializer=init, name='decoder_mean')
            self.x_decoded_mean = Activation('softmax')(Reshape((original_dim,256))(self.decoder_mean(h_decoded)))
        elif loss_type == 'binary':
            self.decoder_mean = Dense(original_dim, activation='sigmoid', kernel_initializer=init, name='decoder_mean')
            self.x_decoded_mean = self.decoder_mean(h_decoded)
        elif loss_type == 'sad' or loss_type == 'ssd':
            self.decoder_mean = Dense(original_dim, kernel_initializer=init, name='decoder_mean')
            self.x_decoded_mean = self.decoder_mean(h_decoded)
        else:
            raise ValueException('Unknown loss type: %s'%loss_type)
    
    def xent_loss(self,labels,x_decoded_mean):
        if self.loss_type == 'categorical':
            xent_loss = K.sum(keras.losses.categorical_crossentropy(labels, x_decoded_mean),axis=-1)
        elif self.loss_type == 'binary':
            xent_loss = self.original_dim * keras.losses.binary_crossentropy(labels,x_decoded_mean)
        elif self.loss_type == 'sad':
            xent_loss = K.sum(K.abs(labels - x_decoded_mean), axis=-1)
        elif self.loss_type == 'ssd':
            xent_loss = K.sum(K.square(labels - x_decoded_mean), axis=-1)
        else:
            raise ValueException('Unknown loss type: %s'%self.loss_type)
        return xent_loss

    def kl_loss(self,labels,x_decoded_mean):
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return kl_loss
    
    def vae_loss(self,labels,x_decoded_mean):
        return 100.*self.xent_loss(labels,x_decoded_mean) + self.kl_loss(labels,x_decoded_mean)
        #return self.xent_loss(labels,x_decoded_mean)

    def get_vae_model(self):
        if self.conditional:
            return Model([self.x,self.h],self.x_decoded_mean)
        else:
            return Model(self.x,self.x_decoded_mean)
    
    def get_encoder_model(self):
        if self.conditional:
            return Model([self.x,self.h], [self.z_mean,self.z_log_var])
        else:
            return Model(self.x, [self.z_mean,self.z_log_var])

    def get_decoder_model(self):
        decoder_input = Input(shape=(self.latent_dim,))
        res = decoder_input
        for i in range(self.num_layers):
            res = self.decoder_h[i](res)
            if self.conditional:
                res = Add()([res,self.decoder_b[i](self.h)])
        _h_decoded = res
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        if self.loss_type == 'categorical':
            _x_decoded_mean = Activation('softmax')(Reshape((self.original_dim,256))(_x_decoded_mean))
        if self.conditional:
            return Model([decoder_input,self.h], _x_decoded_mean)
        else:
            return Model(decoder_input, _x_decoded_mean)

def get_blocks(data,block_size,num_samples,labels=None):
    num_data = len(data)
    data_height = data.shape[1]
    data_width = data.shape[2]
    data_channels = data.shape[3]
    blocks = np.zeros((num_samples,block_size*block_size*data_channels),dtype=data.dtype)
    if labels is not None:
        block_labels = np.zeros((num_samples,labels.shape[1]),dtype=labels.dtype)

    if block_size == data_height and block_size == data_width:
        for n in trange(num_samples, desc='Getting blocks'):
            blocks[n] = data[n].flatten()

    else:
        for n in trange(num_samples,desc='Getting blocks'):
            i = np.random.randint(0,num_data)
            y = np.random.randint(0,data_height-block_size)
            x = np.random.randint(0,data_width-block_size)
            blocks[n] = data[i,y:(y+block_size),x:(x+block_size),:].flatten()

            if labels is not None:
                block_labels[n] = labels[i]

    if labels is not None:
        return blocks, block_labels
    return blocks

def image_to_blocks(im,block_size,flattened=True):
    height = im.shape[0]
    width = im.shape[1]
    channels = im.shape[2]
    num_blocks_y = height/block_size
    num_blocks_x = width/block_size
    if flattened:
        blocks = np.zeros((num_blocks_y*num_blocks_x,block_size*block_size*channels),dtype=im.dtype)
    else:
        blocks = np.zeros((num_blocks_y,num_blocks_x,block_size*block_size*channels),dtype=im.dtype)
    n = 0
    for y in xrange(0,height,block_size):
        for x in xrange(0,width,block_size):
            if flattened:
                blocks[n] = im[y:(y+block_size),x:(x+block_size),:].flatten()
            else:
                blocks[y,x] = im[y:(y+block_size),x:(x+block_size),:].flatten() 
            n = n+1
    return blocks

def probs_to_blocks(pred_probs,usemax=True):
    pred_blocks = np.zeros(pred_probs.shape[0:2],pred_probs.dtype)
    for i in xrange(len(pred_probs)):
        for j in xrange(pred_probs.shape[1]):
            if usemax:
                pred_blocks[i,j] = np.argmax(pred_probs[i,j])
            else:
                pred_blocks[i,j] = np.argmax(np.random.multinomial(1,pred_probs[i,j],1))
    return pred_blocks#/255.

def blocks_to_image(blocks,height,width,channels,block_size,flattened=True):
    num_blocks_y = height/block_size
    num_blocks_x = width/block_size
    im = np.zeros((height,width,channels),dtype=blocks.dtype)
    n = 0
    for y in xrange(0,height,block_size):
        for x in xrange(0,width,block_size):
            if flattened:
                im[y:(y+block_size),x:(x+block_size),:] = np.reshape(blocks[n],(block_size,block_size,channels))
            else:
                im[y:(y+block_size),x:(x+block_size),:] = np.reshape(blocks[y,x],(block_size,block_size,channels))
            n = n+1
    return im

def blocks_to_categorical(x):
    y = np.zeros((len(x),x.shape[1],256),dtype='bool')
    for i in xrange(len(x)):
        y[i] = keras.utils.to_categorical(x[i],256)
    return y


def get_pred_image(im, label):

    blocks = image_to_blocks(im, block_size)
    labels = np.repeat(label, len(blocks), axis=0)

    if conditional:
        encoded, log_var = encoder.predict([blocks, labels])
    else:
        encoded, log_var = encoder.predict(blocks)

    if block_vae.loss_type == 'categorical':
        if conditional:
            pred_probs = generator.predict([encoded, labels])
        else:
            pred_probs = generator.predict(encoded)
        pred_blocks = probs_to_blocks(pred_probs)
    else:
        if conditional:
            pred_blocks = generator.predict([encoded, labels]) + training_mean
        else:
            pred_blocks = generator.predict(encoded) + training_mean

    pred_im = blocks_to_image(pred_blocks, im.shape[0], im.shape[1],
                              im.shape[2], block_size)

    return pred_im

def get_ssim_of_img(im, i):

    mch = False

    label = labels_train[[i]]
    pred_im = get_pred_image(im, label)

    im = im[:, :, 0]
    pred_im = pred_im[:, :, 0]

    if im.shape[-1] > 1 and pred_im.shape[-1] > 1:
        mch = True

    return ssim(im, pred_im, data_range=pred_im.max() - pred_im.min(), multichannel=mch)

def calc_avg_ssim(x_images):

    avg_ssim = 0
    for i in trange(x_images.shape[0], desc='Calculating mean SSIM'):
        im = x_images[i]
        current_ssim = get_ssim_of_img(im,i)
        avg_ssim += current_ssim

    avg_ssim /= x_images.shape[0]

    print "\nThe average SSIM is: " + str(avg_ssim) + "\n"

def calc_nll(im, pred_im, loss_type):
    if loss_type == "binary":
        clipped_pred = np.clip(pred_im,1e-15,1.-1e-15)
        nll = im * np.log(clipped_pred) + (1.-im) * np.log(1.-clipped_pred)
        nll = - np.sum(nll)
    elif loss_type == "sad":
        nll = abs(im - pred_im)
        nll = np.sum(nll)
    else:
        raise ValueException('Unknown loss type %s'%loss_type)

    return nll


# def calc_b_dim(nll):
#     return -((nll/3072)-4.852)/np.log(2.)


def calc_avg_nll(images, block_size, dataset, loss_type):
    overhead = np.log(2) if loss_type == "sad" else 0

    if dataset == "mnist":
        side = 28
        channels = 1
    elif dataset == "cifar10":
        side = 32
        channels = 3
    elif dataset == "lfw":
        side = 128
        channels = 3

    n_images = len(images)
    n = side/block_size

    pre_avg_nll = np.zeros((n_images*n*n,block_size*block_size*channels),dtype=images.dtype)

    for i in trange(len(images), desc='Calculating NLL'):
        im = images[i]
        # label = labels[[i]]

        blocks = image_to_blocks(im, block_size)
        # pred_im = get_pred_image(im, label)
        n_elems = blocks.shape[0]
        for j in range(n_elems):
            pre_avg_nll[n_elems*i+j] = blocks[j]

    # flat_pre = np.expand_dims(pre_avg_nll.flatten(), axis=0)
    res = vae.evaluate(pre_avg_nll, pre_avg_nll, verbose=1)
    xent_nll = res[1] * (n*n)
    kl_nll = res[2] * (n*n)
    avg_nll = xent_nll + kl_nll

    print "\nThe average cross-entropy/SAD loss in nats is: " + str(xent_nll)
    print "\nThe average KL loss in nats is: " + str(kl_nll)
    if dataset == "mnist":
        print "\nThe average NLL in nats is: " + str(avg_nll)
    elif dataset == "cifar10":
        print "\nThe average NLL in bits/dim is: " + str(-((avg_nll/3072)-4.852)/np.log(2.))

def plot_latent(images, block_size, dataset):
    side = images.shape[1]
    channels = images.shape[3]
    n_images = len(images)
    n = side/block_size

    pre_avg_nll = np.zeros((n_images*n*n,block_size*block_size*channels),dtype=images.dtype)

    for i in trange(len(images), desc='Getting test image as blocks'):
        im = images[i]
        # label = labels[[i]]

        blocks = image_to_blocks(im, block_size)
        # pred_im = get_pred_image(im, label)
        n_elems = blocks.shape[0]
        for j in range(n_elems):
            pre_avg_nll[n_elems*i+j] = blocks[j]

    encoded = encoder.predict(pre_avg_nll,verbose=1)[0]
    plt.figure(figsize=(6,6))
    plt.scatter(encoded[:, 0], encoded[:, 1])
    plt.savefig('latent_plot.png')

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

    # if original_dim <> cfg.original_dim:
    #     raise ValueException('Calculated original_dim (%d) does not match configuration file (%d)'%(original_dim,cfg.original_dim))

    #TODO: Each of the following statements can fail when args.<param_name> = 0!
    intermediate_dim = cfg.intermediate_dim if not args.intermediate_dim else args.intermediate_dim
    latent_dim = cfg.latent_dim if not args.latent_dim else args.latent_dim
    num_samples = cfg.num_samples if not args.num_samples else args.num_samples
    num_layers = cfg.num_layers if not args.num_layers else args.num_layers
    loss_type = cfg.vae_loss_type if not args.vae_loss_type else args.vae_loss_type
    batch_size = cfg.batch_size if not args.batch_size else args.batch_size
    epochs = cfg.epochs if not args.epochs else args.epochs
    dataset = cfg.dataset if not args.dataset else args.dataset
    conditional = cfg.conditional if not args.conditional else args.conditional
    epsilon_std = cfg.epsilon_std if not args.epsilon_std else args.epsilon_std

    block_vae_weights = cfg.block_vae_weights
    results_dir = cfg.results_dir

    block_vae_outputs_dir = cfg.get_bvae_out_path()

    print "------------------------------"
    print "Dataset: ", dataset
    print "Block size: ", block_size
    print "Original dim: ", original_dim
    print "Latent dim ", latent_dim
    print "------------------------------"


    # load dataset
    if dataset == 'cifar10':
        print('loading cifar10...')
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
        # select only frogs
        x_train = x_train[(y_train==6).flatten()]
        x_test = x_test[(y_test==6).flatten()]
        y_train = y_train[(y_train==6).flatten()]
        y_test = y_test[(y_test==6).flatten()]
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

    all_labels_train = keras.utils.to_categorical(y_train,num_classes)
    all_labels_test = keras.utils.to_categorical(y_test,num_classes)

        # convert to Lab space
        #for i in trange(len(x_train), desc='converting training images to Lab'):
        #x_train[i] = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2Lab)
        #for i in trange(len(x_test), desc='converting training images to Lab'):
        #x_test[i] = cv2.cvtColor(x_test[i], cv2.COLOR_RGB2Lab)

    if red_only:
        # select only red channel
        x_train = x_train[:,:,:,[0]]
        x_test = x_test[:,:,:,[0]]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #if loss_type == 'binary':
    x_train /= 255.
    x_test /= 255.

    imgs_train = x_train
    imgs_test = x_test

    x_train, labels_train = get_blocks(x_train,block_size,num_samples,all_labels_train)
    x_test, labels_test = get_blocks(x_test,block_size,num_samples,all_labels_test)

    if loss_type == 'categorical':
        y_train = blocks_to_categorical(x_train)#*255)
        y_test = blocks_to_categorical(x_test)#*255)
    else:
        y_train = np.copy(x_train)
        y_test = np.copy(x_test)

    print(x_train.shape)
    print(y_train.shape)

    print('removing means...')
    training_mean = cfg.training_mean
    x_train -= training_mean
    x_test -= training_mean

    block_vae = BlockVAE(original_dim=original_dim,intermediate_dim=intermediate_dim,
                         latent_dim=latent_dim,num_layers=num_layers,loss_type=loss_type,
                          epsilon_std=epsilon_std,conditional=conditional)

    vae = block_vae.get_vae_model()

    print(vae.summary())

    start_time_compile = time.time()
    vae.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss=block_vae.vae_loss, metrics=[block_vae.xent_loss,block_vae.kl_loss])
    elapsed_time_compile = time.time() - start_time_compile

    print "---------------------------------------------"
    print "Elapsed time to compile BlockVAE in seconds: ", elapsed_time_compile
    print "---------------------------------------------"

    if os.path.exists(block_vae_outputs_dir+block_vae_weights):
        vae.load_weights(block_vae_outputs_dir+block_vae_weights)
    else:
        def schedule(epoch):
            step = epoch/50
            lr = cfg.lr * pow(10,-step)
            print('epoch: %d\tlr: %f'%(epoch,lr))
            return lr
        scheduler = keras.callbacks.LearningRateScheduler(schedule)
        plateau = keras.callbacks.ReduceLROnPlateau(monitor='loss',verbose=1,epsilon=0.01)
        stopping = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.001,verbose=1,patience=10)
        checkpoint = keras.callbacks.ModelCheckpoint(block_vae_outputs_dir+block_vae_weights,save_weights_only=True,period=10)
        if os.path.exists('./logs'):
            shutil.rmtree('./logs')
        os.makedirs('./logs')       
        tb = keras.callbacks.TensorBoard()
        if block_vae.loss_type == 'categorical' or block_vae.loss_type == 'binary':
            if conditional:
                inputs = [[x_train,labels_train],y_train]
                val_inputs = [[x_test,labels_test],y_test]
            else:
                inputs = [x_train,y_train]
                val_inputs = [x_test,y_test]
        else:
            if conditional:
                inputs = [[x_train,labels_train],x_train]
                val_inputs = [[x_test,labels_test],x_test]
            else:
                inputs = [x_train,x_train]
                val_inputs = [x_test,x_test]

        start_time_fit = time.time()
        vae.fit(inputs[0],inputs[1],
                validation_data=val_inputs,
                batch_size=batch_size,
                shuffle=True,
                epochs=epochs,
                callbacks=[tb,checkpoint,stopping])
        elapsed_time_fit = time.time() - start_time_fit

        print "---------------------------------------------"
        print "Elapsed time to fit BlockVAE in seconds: ", elapsed_time_fit
        print "---------------------------------------------"

        vae.save_weights(block_vae_outputs_dir+block_vae_weights)

    # build a model to project inputs on the latent space
    encoder = block_vae.get_encoder_model()

    # build a block generator that can sample from the learned distribution
    generator = block_vae.get_decoder_model()

    # if args.dataset == 'mnist' or cfg.dataset == "mnist":
    #     plot_latent(imgs_test, block_size, dataset)
    # else:
    #     plot_latent(imgs_test[::100], block_size, dataset)

    # pass training images through the vae
    avg_time_img1 = 0
    for i in trange(num_classes,desc='predicting training images'):
        ind = np.argmax(all_labels_train[:,i])
        im = imgs_train[ind]
        if red_only:
            imsave(block_vae_outputs_dir+'train_image_%02d.png'%i,im[:,:,0])
        else:
            imsave(block_vae_outputs_dir+'train_image_%02d.png'%i,im)

        start_time_img1 = time.time()
        blocks = image_to_blocks(im,block_size)
        labels = np.repeat(all_labels_train[[ind]],len(blocks),axis=0)
        if conditional:
            encoded, log_var = encoder.predict([blocks,labels])
        else:
            encoded, log_var = encoder.predict(blocks)
        if block_vae.loss_type == 'categorical':
            if conditional:
                pred_probs = generator.predict([encoded,labels])
            else:
                pred_probs = generator.predict(encoded)
            pred_blocks = probs_to_blocks(pred_probs)
        else:
            if conditional:
                pred_blocks = generator.predict([encoded,labels]) + training_mean
            else:
                pred_blocks = generator.predict(encoded) + training_mean
        pred_im = blocks_to_image(pred_blocks,im.shape[0],im.shape[1],im.shape[2],block_size)
        #if loss_type == 'binary':
        pred_im = np.clip(pred_im,0,1) * 255.
        pred_im = pred_im.astype('uint8')
        elapsed_time_img1 = time.time() - start_time_img1

        avg_time_img1 += elapsed_time_img1

        if red_only:
            imsave(block_vae_outputs_dir+'pred_train_image_%02d.png'%i,pred_im[:,:,0])
        else:
            imsave(block_vae_outputs_dir+'pred_train_image_%02d.png'%i,pred_im)
    avg_time_img1 /= num_classes


    # pass testing images through the vae
    avg_time_img2 = 0
    for i in trange(num_classes,desc='predicting testing images'):
        ind = np.argmax(all_labels_test[:,i])
        im = imgs_test[ind]
        if red_only:
            imsave(block_vae_outputs_dir+'test_image_%02d.png'%i,im[:,:,0])
        else:
            imsave(block_vae_outputs_dir+'test_image_%02d.png'%i,im)

        start_time_img2 = time.time()
        blocks = image_to_blocks(im,block_size)
        labels = np.repeat(all_labels_test[[ind]],len(blocks),axis=0)
        if conditional:
            encoded, log_var = encoder.predict([blocks,labels])
        else:
            encoded, log_var = encoder.predict(blocks)
        if block_vae.loss_type == 'categorical':
            if conditional:
                pred_probs = generator.predict([encoded,labels])
            else:
                pred_probs = generator.predict(encoded)
            pred_blocks = probs_to_blocks(pred_probs)
        else:
            if conditional:
                pred_blocks = generator.predict([encoded,labels]) + training_mean
            else:
                pred_blocks = generator.predict(encoded) + training_mean
        pred_im = blocks_to_image(pred_blocks,im.shape[0],im.shape[1],im.shape[2],block_size)
        #if loss_type == 'binary':
        pred_im = np.clip(pred_im,0.,1.) * 255.
        pred_im = pred_im.astype('uint8')
        elapsed_time_img2 = time.time() - start_time_img2

        avg_time_img2 += elapsed_time_img2

        if red_only:
            imsave(block_vae_outputs_dir+'pred_test_image_%02d.png'%i,pred_im[:,:,0])
        else:
            imsave(block_vae_outputs_dir+'pred_test_image_%02d.png'%i,pred_im)

    avg_time_img2 /= num_classes

    # sample a random image
    avg_time_sampled = 0
    for i in xrange(num_classes):
        start_time_sampled = time.time()
        z_sample = np.random.normal(size=(len(blocks),block_vae.latent_dim))
        labels_sample = np.zeros((len(z_sample),num_classes),dtype=bool)
        labels_sample[:,0] = True
        if block_vae.loss_type == 'categorical':
            if conditional:
                probs = generator.predict([z_sample,labels_sample])
            else:
                probs = generator.predict(z_sample)
            blocks = probs_to_blocks(probs)
        else:
            if conditional:
                blocks = generator.predict([z_sample,labels_sample]) + training_mean
            else:
                blocks = generator.predict(z_sample) + training_mean

        elapsed_time_sampled = time.time() - start_time_sampled

        pred_im = blocks_to_image(blocks,im.shape[0],im.shape[1],im.shape[2],block_size)
        #if loss_type == 'binary':
        pred_im = np.clip(pred_im,0,1) * 255.
        pred_im = pred_im.astype('uint8')

        if red_only:
            imsave(block_vae_outputs_dir+'sampled_image_%02d.png'%i,pred_im[:,:,0])
        else:
            imsave(block_vae_outputs_dir+'sampled_image_%02d.png'%i,pred_im)

        avg_time_sampled += elapsed_time_sampled

    avg_time_sampled /= num_classes

    print "---------------------------------------------"
    print "BlockVAE generation times"
    print "---------------------------------------------"
    print "Elapsed time image 1: ", avg_time_img1
    print "Elapsed time image 2: ", avg_time_img2
    print "Elapsed time sampled: ", avg_time_sampled
    print "---------------------------------------------"

    if args.calc_ssim:
        calc_avg_ssim(imgs_test)

    if args.calc_nll:
        calc_avg_nll(imgs_test, block_size, dataset, loss_type)

