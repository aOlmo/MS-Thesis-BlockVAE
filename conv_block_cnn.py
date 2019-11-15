#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os
import sys
import argparse
import configparser
from datetime import datetime
import pytz
from distutils.util import strtobool
import numpy as np
import keras
from keras.datasets import mnist, cifar10

from layers import PixelCNN
from utils import Utils

from conv_block_vae import BlockVAE
from tqdm import trange

from configs import Configs

import time

def generate_bottom_half(X_pred, labels, conditional, block_vae, pixelcnn):
    nb_images = len(X_pred)
    input_size = X_pred.shape[1:3]
    ## generate encoded images block by block
    for i in range(input_size[0] / 2, input_size[0]):
        for j in range(input_size[1]):
            x = X_pred

            if conditional:
                next_X_pred = pixelcnn.model.predict([x,labels])
            else:
                next_X_pred = pixelcnn.model.predict(x)
            samp = next_X_pred[:, i, j, :]
            #X_pred[:, i, j, :] = samp
            noise = np.random.randn(nb_images,block_vae.latent_dim)
            X_pred[:,i,j,:] = samp+noise*0.1

    return X_pred

def decode_images_and_predict(x, block_vae, decoder_model, cfg, img_desc):
    block_cnn_outputs_dir = cfg.get_bcnn_out_path()
    block_size = cfg.block_size

    decoded_x = decoder_model.predict(x)
    for i in xrange(len(x)):
        im_pred = decoded_x[i]
        imsave(block_cnn_outputs_dir + 'gen_'+img_desc+'_image_%02d.png' % i, np.squeeze(im_pred))


def train(argv=None):
    ''' train Block Gated PixelCNN model 
    Usage:
    	python block_cnn.py -c sample_train.cfg        : training example using configfile
    	python block_cnn.py --option1 hoge ...         : train with command-line options
        python block_cnn.py -c test.cfg --opt1 hoge... : overwrite config options with command-line options
    '''


    ### parsing arguments from command-line or config-file ###
    if argv is None:
        argv = sys.argv

    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {}

    # if args.conf_file:
        # config = configparser.SafeConfigParser()
        # config.read([args.conf_file])
        # defaults.update(dict(config.items("General")))

    cfg = Configs(args.conf_file) if args.conf_file else Configs()

    original_dim = cfg.original_dim
    intermediate_dim = cfg.intermediate_dim
    latent_dim = cfg.latent_dim
    block_size = cfg.block_size
    dataset = cfg.dataset
    red_only = cfg.red_only

    epsilon_std = cfg.epsilon_std

    block_cnn_weights = cfg.block_cnn_weights
    block_cnn_nb_epoch = cfg.block_cnn_nb_epoch
    block_cnn_batch_size = cfg.block_cnn_batch_size

    block_vae_weights = cfg.block_vae_weights

    block_cnn_outputs_dir = cfg.get_bcnn_out_path()
    block_vae_outputs_dir = cfg.get_bvae_out_path()

    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**defaults)
    parser.add_argument("--nb_epoch", help="Number of epochs [Required]", type=int, metavar="INT")
    parser.add_argument("--nb_images", help="Number of images to generate",type=int, metavar="INT")
    parser.add_argument("--batch_size", help="Minibatch size", type=int, metavar="INT")
    parser.add_argument("--conditional", help="model the conditional distribution p(x|h) (default:False)", type=str, metavar="BOOL")
    parser.add_argument("--nb_pixelcnn_layers", help="Number of PixelCNN Layers (exept last two ReLu layers)", metavar="INT")
    parser.add_argument("--nb_filters", help="Number of filters for each layer", metavar="INT")
    parser.add_argument("--filter_size_1st", help="Filter size for the first layer. (default: (7,7))", metavar="INT,INT")
    parser.add_argument("--filter_size", help="Filter size for the subsequent layers. (default: (3,3))", metavar="INT,INT")
    parser.add_argument("--optimizer", help="SGD optimizer (default: adadelta)", type=str, metavar="OPT_NAME")
    parser.add_argument("--es_patience", help="Patience parameter for EarlyStopping", type=int, metavar="INT")
    parser.add_argument("--save_root", help="Root directory which trained files are saved (default: /tmp/pixelcnn)", type=str, metavar="DIR_PATH")
    parser.add_argument("--timezone", help="Trained files are saved in save_root/YYYYMMDDHHMMSS/ (default: Asia/Tokyo)", type=str, metavar="REGION_NAME")
    parser.add_argument("--save_best_only", help="The latest best model will not be overwritten (default: False)", type=str, metavar="BOOL")

    args = parser.parse_args(remaining_argv)

    conditional = strtobool(args.conditional) if args.conditional else False

    ### load dataset ###
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        # add dimension for channels
        x_train = np.expand_dims(x_train,axis=-1)
        x_test = np.expand_dims(x_test,axis=-1)
    if red_only:
        # select only red channel
        x_train = x_train[:,:,:,[0]]
        x_test = x_test[:,:,:,[0]]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    h_train = keras.utils.to_categorical(y_train,num_classes)
    h_test = keras.utils.to_categorical(y_test,num_classes)
        
    imgs_train = x_train
    imgs_test = x_test

    num_blocks_y = x_train.shape[1]/block_size
    num_blocks_x = x_train.shape[2]/block_size
    num_blocks = num_blocks_y * num_blocks_x
    num_channels = imgs_train.shape[3]

    ### encoded block image size ###
    input_size = (num_blocks_y, num_blocks_x)

    utils = Utils()

    # run blocks through pre-trained encoder
    block_vae = BlockVAE(x_train.shape[1:4], block_size, intermediate_dim, latent_dim)
    vae_model = block_vae.get_vae_model()
    encoder_model = block_vae.get_encoder_model()
    decoder_model = block_vae.get_decoder_model()
    vae_model.load_weights(block_vae_outputs_dir+block_vae_weights)
    

    ### build PixelCNN model ###
    model_params = {}
    model_params['input_size'] = input_size
    model_params['nb_channels'] = block_vae.latent_dim
    model_params['conditional'] = conditional
    if conditional:
        model_params['latent_dim'] = num_classes
    if args.nb_pixelcnn_layers:
        model_params['nb_pixelcnn_layers'] = int(args.nb_pixelcnn_layers)
    if args.nb_filters:
        model_params['nb_filters'] = int(args.nb_filters)
    if args.filter_size_1st:
        model_params['filter_size_1st'] = tuple(map(int, args.filter_size_1st.split(',')))
    if args.filter_size:
        model_params['filter_size'] = tuple(map(int, args.filter_size.split(',')))
    if args.optimizer:
        model_params['optimizer'] = args.optimizer
    if args.es_patience:
        model_params['es_patience'] = int(args.patience)
    if args.save_best_only:
        model_params['save_best_only'] = strtobool(args.save_best_only)

    save_root = args.save_root if args.save_root else '/tmp/pixelcnn_mnist'
    timezone = args.timezone if args.timezone else 'Asia/Tokyo'
    current_datetime = datetime.now(pytz.timezone(timezone)).strftime('%Y%m%d_%H%M%S')
    save_root = os.path.join(save_root, current_datetime)
    model_params['save_root'] = save_root

    if not os.path.exists(save_root):
        os.makedirs(save_root)


    pixelcnn = PixelCNN(**model_params)
    pixelcnn.build_model()

    if not os.path.exists(block_cnn_outputs_dir+block_cnn_weights):

        # NOTE: Now it is compulsory to add the nb_epoch and
        # batch_size variables in the configuration file as
        # 'block_cnn_nb_epoch' and 'block_cnn_batch_size' respectively
        nb_epoch = block_cnn_nb_epoch
        batch_size = block_cnn_batch_size

        # try:
        #     nb_epoch = int(args.nb_epoch)
        #     batch_size = int(args.batch_size)
        # except:
        #     sys.exit("Error: {--nb_epoch, --batch_size} must be specified.")


        pixelcnn.print_train_parameters(save_root)
        pixelcnn.export_train_parameters(save_root)
        with open(os.path.join(save_root, 'parameters.txt'), 'a') as txt_file:
            txt_file.write('########## other options ##########\n')
            txt_file.write('nb_epoch\t: %s\n' % nb_epoch)
            txt_file.write('batch_size\t: %s\n' % batch_size)
            txt_file.write('\n')

        pixelcnn.model.summary()

        # encode images using VAE
        print('Encoding blocks...')
        encoded_blocks_train = encoder_model.predict(x_train,verbose=1,batch_size=batch_size)[0]
        encoded_blocks_test = encoder_model.predict(x_test,verbose=1,batch_size=batch_size)[0]

        train_params = {}
        if conditional:
            train_params['x'] = [encoded_blocks_train,h_train]
            train_params['validation_data'] = ([encoded_blocks_test,h_test],encoded_blocks_test)
        else:
            train_params['x'] = encoded_blocks_train
            train_params['validation_data'] = (encoded_blocks_test,encoded_blocks_test)
        train_params['y'] = encoded_blocks_train
        train_params['nb_epoch'] = nb_epoch
        train_params['batch_size'] = batch_size
        train_params['shuffle'] = True

        start_time = time.time()
        pixelcnn.fit(**train_params)
        elapsed_time = time.time() - start_time

        print '------------------------------------------------'
        print 'Elapsed time: ' + str(elapsed_time)

        pixelcnn.model.save_weights(block_cnn_outputs_dir+block_cnn_weights)
    else:
        pixelcnn.model.load_weights(block_cnn_outputs_dir+block_cnn_weights)

    ## prepare zeros array
    nb_images = int(args.nb_images) if args.nb_images else 8
    batch_size = int(args.batch_size) if args.batch_size else nb_images
    #X_pred = np.zeros((nb_images, input_size[0], input_size[1], block_vae.latent_dim))

    # encode training images using BlockVAE 
    X_pred = encoder_model.predict(x_train[0:nb_images])[0]
    # generate encode bottom half of images block by block
    X_pred = generate_bottom_half(X_pred, h_train[0:nb_images], conditional, block_vae, pixelcnn)
    # decode encoded images
    decode_images_and_predict(X_pred, block_vae, decoder_model, cfg, 'train')

    # encode testing images using BlockVAE
    X_pred = encoder_model.predict(x_test[0:nb_images])[0]
    # generate encode bottom half of images block by block
    X_pred = generate_bottom_half(X_pred, h_test[0:nb_images], conditional, block_vae, pixelcnn)
    # decode encoded images
    decode_images_and_predict(X_pred, block_vae, decoder_model, cfg, 'test')

    # randomly sample images
    X_pred = np.zeros((num_classes,num_blocks_y,num_blocks_x,block_vae.latent_dim))
    h_pred = np.arange(num_classes)
    h_pred = keras.utils.to_categorical(h_pred,num_classes)

    ### generate encoded images block by block
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            if conditional:
                x = [X_pred,h_pred]
            else:
                x = X_pred

            next_X_pred = pixelcnn.model.predict(x, batch_size)
            samp = next_X_pred[:,i,j,:]
            #X_pred[:,i,j,:] = samp
            noise = np.random.randn(num_classes,block_vae.latent_dim)
            X_pred[:,i,j,:] = samp+noise*0.1

    # decode encoded images
    decode_images_and_predict(X_pred, block_vae, decoder_model, cfg, 'sampled')


if __name__ == '__main__':
    sys.exit(train())
