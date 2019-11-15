#!/usr/bin/env python
import numpy as np
np.random.seed(0)

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
import keras
from keras.datasets import mnist, cifar10

from layers import PixelCNN
from utils import Utils

from block_vae import BlockVAE, image_to_blocks, blocks_to_image
from tqdm import trange

from configs import Configs

import time

import h5py


def calc_avg_nll(conditional, labels, encoder_model, block_vae, input_size, pixelcnn, images, block_size, dataset):
    avg_nll = 0

    side = 28 if dataset == "mnist" else 32
    channels = 1 if dataset == "mnist" else 3
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

    encoded_blocks = encode_blocks_with_VAE(pre_avg_nll, encoder_model, len(images), block_vae, input_size)

    if conditional:
        L = pixelcnn.model.evaluate([encoded_blocks, labels], encoded_blocks, verbose=0)
    else:
        L = pixelcnn.model.evaluate(encoded_blocks, encoded_blocks, verbose=0)
    avg_nll = (n/2)*np.log(2*np.pi) + L/20

    if dataset == "mnist":
        print "\nThe average NLL in nats is: " + str(avg_nll)
    elif dataset == "cifar10":
        print "\nThe average NLL in nats is: " + str(avg_nll) + " and in bits/dim is: " + str(-((avg_nll/3072)-4.852)/np.log(2.))



def encode_blocks_with_VAE(blocks, encoder_model, nb_images, block_vae, input_size):
    n_blocks_side = input_size[0]
    n_blocks_img = input_size[0] * input_size[1]

    results = encoder_model.predict(blocks)[0]
    X_pred = np.zeros((nb_images, n_blocks_side, n_blocks_side, block_vae.latent_dim))
    for i in xrange(nb_images):
        X_pred[i] = results[i * n_blocks_img:(i + 1) * n_blocks_img].reshape(n_blocks_side,
                                                                             n_blocks_side, -1)
    return X_pred

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
            noise = np.clip(noise*0.01,-0.02,0.02)
            X_pred[:,i,j,:] = samp+noise

    return X_pred

def decode_images_and_predict(x, block_vae, decoder_model, cfg, img_desc):
    block_cnn_outputs_dir = cfg.get_bcnn_out_path()
    block_size = cfg.block_size

    num_enc_blocks_side = x.shape[1]
    num_enc_blocks_img = num_enc_blocks_side*num_enc_blocks_side
    img_size = block_size * num_enc_blocks_side

    if cfg.red_only:
        num_channels = 1
    else:
        num_channels = 3

    blocks_pred = np.zeros((len(x)*num_enc_blocks_img, block_vae.latent_dim), dtype='float32')
    start_time = time.time()
    for i in xrange(len(x)):
        blocks_pred[i*num_enc_blocks_img: (i+1)*num_enc_blocks_img, :] = \
                        x[i].reshape((num_enc_blocks_img, block_vae.latent_dim))

    decoded_blocks = decoder_model.predict(blocks_pred)
    final_time = time.time() - start_time

    for i in xrange(len(x)):
        im_pred = blocks_to_image(decoded_blocks[i * num_enc_blocks_img: (i+1)*num_enc_blocks_img, :],
                        img_size, img_size, num_channels, block_size)
        if num_channels == 1:
            im_pred = np.squeeze(im_pred,axis=-1)
        imsave(block_cnn_outputs_dir + 'gen_'+img_desc+'_image_%02d.png' % i, im_pred)

    return final_time



def get_images_to_blocks(x, nb_images, block_size):

    # NOTE: in this implementation, we assume that the blocks
    # are squared and have the same number of pixels per row and column!

    dim = x.shape[1]
    num_channels = x.shape[3]
    n_blocks_side = dim / block_size
    n_blocks_img = n_blocks_side*n_blocks_side
    total_num_blocks = nb_images*n_blocks_img
    flat_dims_per_block = block_size*block_size*num_channels

    blocks = np.zeros((total_num_blocks, flat_dims_per_block), dtype='float32')
    for i in xrange(nb_images):
        blocks[i*n_blocks_img: (i+1)*n_blocks_img] = image_to_blocks(x[i], block_size)

    return blocks


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
    num_layers = cfg.num_layers
    loss_type = cfg.vae_loss_type
    block_size = cfg.block_size
    dataset = cfg.dataset
    red_only = cfg.red_only

    epsilon_std = cfg.epsilon_std
    
    gated = cfg.gated

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
    parser.add_argument("--calc_nll", help="Calculate the average NLL of the images (default: False)", type=int, metavar="INT")

    args = parser.parse_args(remaining_argv)

    conditional = strtobool(args.conditional) if args.conditional else False

    print "------------------------------"
    print "Dataset: ", dataset
    print "Block size: ", block_size
    print "Original dim: ", original_dim
    print "Latent dim ", latent_dim
    print "------------------------------"

    ### load dataset ###
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10

        # select only frogs
        num_classes = 1
        x_train = x_train[(y_train==6).flatten()]
        y_train = y_train[(y_train==6).flatten()]
        x_test = x_test[(y_test==6).flatten()]
        y_test = y_test[(y_test==6).flatten()]
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        # add dimension for channels
        x_train = np.expand_dims(x_train,axis=-1)
        # x_train = x_train[:10000]
        x_test = np.expand_dims(x_test,axis=-1)
        # x_test = x_test[:1500]
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
    if red_only:
        # select only red channel
        x_train = x_train[:,:,:,[0]]
        x_test = x_test[:,:,:,[0]]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    if num_classes > 1:
        h_train = keras.utils.to_categorical(y_train,num_classes)
        h_test = keras.utils.to_categorical(y_test,num_classes)
    else:
        h_train = np.copy(y_train)
        h_test = np.copy(y_test)
        
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
    block_vae = BlockVAE(original_dim, intermediate_dim, latent_dim, num_layers, loss_type, epsilon_std)
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
    if gated:
        model_params['gated'] = strtobool(gated)
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

        # get image blocks for all images
        blocks_train = np.zeros((len(x_train)*num_blocks_y*num_blocks_x,block_size*block_size*num_channels),dtype='float32')
        blocks_test = np.zeros((len(x_test)*num_blocks_y*num_blocks_x,block_size*block_size*num_channels),dtype='float32')
        for i in trange(len(x_train),desc='getting training image blocks'):
            blocks_train[i*num_blocks:(i+1)*num_blocks] = image_to_blocks(x_train[i],block_size)
        for i in trange(len(x_test),desc='getting testing image blocks'):
            blocks_test[i*num_blocks:(i+1)*num_blocks] = image_to_blocks(x_test[i],block_size)

        # encode blocks using VAE
        print('Encoding blocks...')
        results = encoder_model.predict(blocks_train,verbose=1,batch_size=batch_size)[0]
        encoded_blocks_train = np.zeros((len(x_train),num_blocks_y,num_blocks_x,block_vae.latent_dim))
        for i in xrange(len(x_train)):
            encoded_blocks_train[i] = results[i*num_blocks:(i+1)*num_blocks].reshape(num_blocks_y,num_blocks_x,-1)

        results = encoder_model.predict(blocks_test,verbose=1,batch_size=batch_size)[0]
        encoded_blocks_test = np.zeros((len(x_test),num_blocks_y,num_blocks_x,block_vae.latent_dim))
        h_test = np.zeros((len(x_test),num_classes))
        for i in xrange(len(x_test)):
            encoded_blocks_test[i] = results[i*num_blocks:(i+1)*num_blocks].reshape(num_blocks_y,num_blocks_x,-1)

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
        print 'Elapsed time in BlockCNN: ' + str(elapsed_time)

        pixelcnn.model.save_weights(block_cnn_outputs_dir+block_cnn_weights)
    else:
        pixelcnn.model.load_weights(block_cnn_outputs_dir+block_cnn_weights)

    ## prepare zeros array
    nb_images = int(args.nb_images) if args.nb_images else 8
    batch_size = int(args.batch_size) if args.batch_size else nb_images
    #X_pred = np.zeros((nb_images, input_size[0], input_size[1], block_vae.latent_dim))

    # get image blocks for all images
    blocks_train = get_images_to_blocks(x_train, nb_images, block_size)
    blocks_test = get_images_to_blocks(x_test, nb_images, block_size)
    
    # write out ground truth images
    # for i in range(nb_images):
    #     imsave(block_cnn_outputs_dir + 'train_image_%02d.png' % i, x_train[i])
    #     imsave(block_cnn_outputs_dir + 'test_image_%02d.png' % i, x_test[i])

    for i in range(10):
        # encode training images using BlockVAE 
        X_pred = encode_blocks_with_VAE(blocks_train, encoder_model, nb_images, block_vae, input_size)
        # generate encode bottom half of images block by block
        X_pred = generate_bottom_half(X_pred, h_train[0:nb_images], conditional, block_vae, pixelcnn)
        # decode encoded images
        elapsed_time_img1 = decode_images_and_predict(X_pred, block_vae, decoder_model, cfg, 'train%d'%i)

    for i in range(10):
        # encode testing images using BlockVAE
        X_pred = encode_blocks_with_VAE(blocks_test, encoder_model, nb_images, block_vae, input_size)
        # generate encode bottom half of images block by block
        X_pred = generate_bottom_half(X_pred, h_test[0:nb_images], conditional, block_vae, pixelcnn)
        # decode encoded images
        elapsed_time_img2 = decode_images_and_predict(X_pred, block_vae, decoder_model, cfg, 'test%d'%i)


    start_time_sampled_overhead = time.time()
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
            noise = np.clip(noise*0.01,-0.02,0.02)
            X_pred[:,i,j,:] = samp+noise

    elapsed_time_sampled_overhead = time.time() - start_time_sampled_overhead
    # decode encoded images
    elapsed_time_sampled = decode_images_and_predict(X_pred, block_vae, decoder_model, cfg, 'sampled')
    elapsed_time_sampled += elapsed_time_sampled_overhead

    print "---------------------------------------------"
    print "BlockCNN generation times"
    print "---------------------------------------------"
    print "Elapsed time image 1: ", elapsed_time_img1
    print "Elapsed time image 2: ", elapsed_time_img2
    print "Elapsed time sampled: ", elapsed_time_sampled
    print "---------------------------------------------"

    if args.calc_nll:
        calc_avg_nll(conditional, h_test, encoder_model, block_vae, input_size, pixelcnn, imgs_test, block_size, dataset)


if __name__ == '__main__':
    sys.exit(train())
