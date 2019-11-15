import os
import shutil

from keras.layers import Input, Dense, Add, Multiply, Dropout
from keras.layers.core import Activation, Reshape, Lambda
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from keras.layers import Conv2D, ZeroPadding2D

from keras import backend as K

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

import tensorflow as tf
from tensorflow.python.client import timeline



class GatedCNN(object):
    ''' Convolution layer with gated activation unit. '''

    def __init__(
        self,
        nb_filters,
        stack_name,
        v_map=None,
        h=None,
        crop_right=False,
        **kwargs):
        '''
        Args:
            nb_filters (int)         : Number of the filters (feature maps)
            stack_name (str)		: 'vertical' or 'horizontal'
            v_map (numpy.ndarray)   : Vertical maps if feeding into horizontal stack. (default:None)
            h (numpy.ndarray)       : Latent vector to model the conditional distribution p(x|h) (default:None)
            crop_right (bool)       : if True, crop rightmost of the feature maps (mask A, introduced in [https://arxiv.org/abs/1601.06759] )
        '''
        self.nb_filters = nb_filters
        self.stack_name = stack_name
        self.v_map = v_map
        self.h = h
        self.crop_right = crop_right

    @staticmethod
    def _crop_right(x):
        x_shape = K.int_shape(x)
        return x[:,:,:x_shape[2]-1,:]


    def __call__(self, xW, layer_idx):
        '''calculate gated activation maps given input maps '''

        # apply Dropout to xW
        #xW = Dropout(0.5)(xW)

        if self.stack_name == 'vertical':
            stack_tag = 'v'
        elif self.stack_name == 'horizontal':
            stack_tag = 'h'

        if self.crop_right:
            xW = Lambda(self._crop_right, name='h_crop_right_'+str(layer_idx))(xW)

        if self.v_map is not None:
            xW = Add(name='h_merge_v_'+str(layer_idx))([xW,self.v_map])
        
        if self.h is not None:
            hV = Dense(2*self.nb_filters, name=stack_tag+'_dense_latent_'+str(layer_idx))(self.h)
            hV = Reshape((1, 1, 2*self.nb_filters), name=stack_tag+'_reshape_latent_'+str(layer_idx))(hV)
            xW = Lambda(lambda x: x[0]+x[1], name=stack_tag+'_merge_latent_'+str(layer_idx))([xW,hV])

        xW_f = Lambda(lambda x: x[:,:,:,:self.nb_filters], name=stack_tag+'_Wf_'+str(layer_idx))(xW)
        xW_g = Lambda(lambda x: x[:,:,:,self.nb_filters:], name=stack_tag+'_Wg_'+str(layer_idx))(xW)

        xW_f = Lambda(lambda x: K.tanh(x), name=stack_tag+'_tanh_'+str(layer_idx))(xW_f)
        xW_g = Lambda(lambda x: K.sigmoid(x), name=stack_tag+'_sigmoid_'+str(layer_idx))(xW_g)

        res = Multiply(name=stack_tag+'_merge_gate_'+str(layer_idx))([xW_f, xW_g])
        #print(type(res), K.int_shape(res), hasattr(res, '_keras_history'))
        return res

class ReLUCNN(object):
    ''' Convolution layer with ReLU activation unit. '''

    def __init__(
        self,
        nb_filters,
        stack_name,
        v_map=None,
        h=None,
        crop_right=False,
        **kwargs):
        '''
        Args:
            nb_filters (int)         : Number of the filters (feature maps)
            stack_name (str)		: 'vertical' or 'horizontal'
            v_map (numpy.ndarray)   : Vertical maps if feeding into horizontal stack. (default:None)
            h (numpy.ndarray)       : Latent vector to model the conditional distribution p(x|h) (default:None)
            crop_right (bool)       : if True, crop rightmost of the feature maps (mask A, introduced in [https://arxiv.org/abs/1601.06759] )
        '''
        self.nb_filters = nb_filters
        self.stack_name = stack_name
        self.v_map = v_map
        self.h = h
        self.crop_right = crop_right

    @staticmethod
    def _crop_right(x):
        x_shape = K.int_shape(x)
        return x[:,:,:x_shape[2]-1,:]


    def __call__(self, xW, layer_idx):
        '''calculate activation maps given input maps '''

        if self.stack_name == 'vertical':
            stack_tag = 'v'
        elif self.stack_name == 'horizontal':
            stack_tag = 'h'

        if self.crop_right:
            xW = Lambda(self._crop_right, name='h_crop_right_'+str(layer_idx))(xW)

        if self.v_map is not None:
            xW = Add(name='h_merge_v_'+str(layer_idx))([xW,self.v_map])
        
        if self.h is not None:
            hV = Dense(self.nb_filters, name=stack_tag+'_dense_latent_'+str(layer_idx))(self.h)
            hV = Reshape((1, 1, self.nb_filters), name=stack_tag+'_reshape_latent_'+str(layer_idx))(hV)
            xW = Lambda(lambda x: x[0]+x[1], name=stack_tag+'_merge_latent_'+str(layer_idx))([xW,hV])

        res = Activation('relu', name=stack_tag+'_relu_latent_'+str(layer_idx))(xW)
        return res

class PixelCNN(object):
    ''' Keras implementation of (conditional) Gated PixelCNN model '''
    def __init__(
        self,
        input_size,
        nb_channels=3,
        conditional=False,
        latent_dim=10,
        nb_pixelcnn_layers=3,
        nb_filters=128,
        gated=True,
        filter_size_1st=(7,7),
        filter_size=(3,3),
        optimizer='adam',
        es_patience=100,
        save_root='.',
        checkpoint_path='weights.h5',
        save_best_only=False,
        **kwargs):
        '''
        Args:
            input_size ((int,int))      : (height, width) pixels of input images
            nb_channels (int)           : Number of channels for input images. (1 for grayscale images, 3 for color images)
            conditional (bool)          : if True, use latent vector to model the conditional distribution p(x|h) (default:False)
            latent_dim (int)            : (if conditional==True,) Dimensions for latent vector.
            nb_pixelcnn_layers (int)    : Number of layers (except last two ReLu layers). (default:13)
            nb_filters (int)            : Number of filters (feature maps) for each layer. (default:128)
            gated (bool)                : if True, use gated activations instead of RelU
            filter_size_1st ((int, int)): Kernel size for the first layer. (default: (7,7))
            filter_size ((int, int))    : Kernel size for the subsequent layers. (default: (3,3))
            optimizer (str)             : SGD optimizer (default: 'adadelta')
            es_patience (int)           : Number of epochs with no improvement after which training will be stopped (EarlyStopping)
            save_root (str)             : Root directory to which {trained model file, parameter.txt, tensorboard log file} are saved
            save_best_only (bool)       : if True, the latest best model will not be overwritten (default: False)
        '''
        K.set_image_dim_ordering('tf')

        self.input_size = input_size
        self.conditional = conditional
        self.latent_dim = latent_dim
        self.nb_pixelcnn_layers = nb_pixelcnn_layers
        self.nb_filters = nb_filters
        self.gated = gated
        self.filter_size_1st = filter_size_1st
        self.filter_size = filter_size
        self.nb_channels = nb_channels
        #self.loss = 'mean_squared_error'
        self.loss = lambda y,x:K.sum(K.square(K.batch_flatten(y)-K.batch_flatten(x)),axis=-1)*10.
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.save_best_only = save_best_only

        tensorboard_dir = './block_cnn_logs' #os.path.join(save_root, 'pixelcnn-tensorboard')
        if os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
        os.makedirs(tensorboard_dir)
        #checkpoint_path = os.path.join(save_root, 'pixelcnn-weights.{epoch:02d}-{val_loss:.4f}.hdf5')
        self.tensorboard = TensorBoard(log_dir=tensorboard_dir)
        ### "save_weights_only=False" causes error when exporting model architecture. (json or yaml)
        self.checkpointer = ModelCheckpoint(filepath=checkpoint_path, period=10, verbose=0, save_weights_only=True, save_best_only=save_best_only)
        self.earlystopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0, mode='auto')
    

    def _masked_conv(self, x, filter_size, stack_name, layer_idx, mask_type='B'):
        my_nb_filters = 2*self.nb_filters if self.gated else self.nb_filters
        if stack_name == 'vertical':
            res = ZeroPadding2D(padding=((filter_size[0]//2, 0), (filter_size[1]//2, filter_size[1]//2)), name='v_pad_'+str(layer_idx))(x)
            res = Conv2D(my_nb_filters, (filter_size[0]//2+1, filter_size[1]), padding='valid', name='v_conv_'+str(layer_idx))(res)
        elif stack_name == 'horizontal':
            res = ZeroPadding2D(padding=((0, 0), (filter_size[1]//2, 0)), name='h_pad_'+str(layer_idx))(x)
            if mask_type == 'A':
                res = Conv2D(my_nb_filters, (1, filter_size[1]//2), padding='valid', name='h_conv_'+str(layer_idx))(res)
            else:
                res = Conv2D(my_nb_filters, (1, filter_size[1]//2+1), padding='valid', name='h_conv_'+str(layer_idx))(res)

        return res


    @staticmethod
    def _shift_down(x):
        x_shape = K.int_shape(x)
        x = ZeroPadding2D(padding=((1,0),(0,0)))(x)
        x = Lambda(lambda x: x[:,:x_shape[1],:,:])(x)
        return x

    def _feed_v_map(self, x, layer_idx):
        ### shifting down feature maps
        x = Lambda(self._shift_down, name='v_shift_down'+str(layer_idx))(x)
        my_nb_filters = 2*self.nb_filters if self.gated else self.nb_filters
        x = Conv2D(my_nb_filters, (1, 1), padding='valid', name='v_1x1_conv_'+str(layer_idx))(x)
        return x


    def _build_layers(self, x, h=None):
        ''' Whole architecture of (conditional) Gated PixelCNN model '''
        # set latent vector
        self.h = h

        # first PixelCNN layer
        ### (kxk) masked convolution can be achieved by (k//2+1, k) convolution and padding.
        v_masked_map = self._masked_conv(x, self.filter_size_1st, 'vertical', 0)
        ### (i-1)-th vertical activation maps into the i-th hirizontal stack. (if i==0, vertical activation maps == input images)
        v_feed_map = self._feed_v_map(v_masked_map, 0)
        if self.gated:
            v_stack_out = GatedCNN(self.nb_filters, 'vertical', v_map=None, h=self.h)(v_masked_map, 0)
        else:
            v_stack_out = ReLUCNN(self.nb_filters, 'vertical', v_map=None, h=self.h)(v_masked_map, 0)
        ### (1xk) masked convolution can be achieved by (1 x k//2+1) convolution and padding.
        h_masked_map = self._masked_conv(x, self.filter_size_1st, 'horizontal', 0, 'A')
        ### Mask A is applied to the first layer (achieved by cropping), and v_feed_maps are merged. 
        if self.gated:
            h_stack_out = GatedCNN(self.nb_filters, 'horizontal', v_map=v_feed_map, h=self.h, crop_right=True)(h_masked_map, 0)
        else:
            h_stack_out = ReLUCNN(self.nb_filters, 'horizontal', v_map=v_feed_map, h=self.h, crop_right=True)(h_masked_map, 0)
        ### not residual connection in the first layer.
        h_stack_out = Conv2D(self.nb_filters, (1, 1), padding='valid', name='h_1x1_conv_0')(h_stack_out)

        # subsequent PixelCNN layers
        for i in range(1, self.nb_pixelcnn_layers):
            v_masked_map = self._masked_conv(v_stack_out, self.filter_size, 'vertical', i)
            v_feed_map = self._feed_v_map(v_masked_map, i)
            if self.gated:
                v_stack_out = GatedCNN(self.nb_filters, 'vertical', v_map=None, h=self.h)(v_masked_map, i)
            else:
                v_stack_out = ReLUCNN(self.nb_filters, 'vertical', v_map=None, h=self.h)(v_masked_map, i)
            ### for residual connection
            h_stack_out_prev = h_stack_out
            h_masked_map = self._masked_conv(h_stack_out, self.filter_size, 'horizontal', i)
            ### Mask B is applied to the subsequent layers.
            if self.gated:
                h_stack_out = GatedCNN(self.nb_filters, 'horizontal', v_map=v_feed_map, h=self.h)(h_masked_map, i)
            else:
                h_stack_out = ReLUCNN(self.nb_filters, 'horizontal', v_map=v_feed_map, h=self.h)(h_masked_map, i)
            h_stack_out = Conv2D(self.nb_filters, (1, 1), padding='valid', name='h_1x1_conv_'+str(i))(h_stack_out)
            ### residual connection
            h_stack_out = Add(name='h_residual_'+str(i))([h_stack_out, h_stack_out_prev])

        # Makes the image more blurry
        # for i in range(2):
        #     h_stack_out = Dense(256, activation='relu', name='testing'+str(i))(h_stack_out)

        # (1x1) convolution layers (2 layers)
        for i in range(2):
            h_stack_out = Conv2D(self.nb_filters, (1, 1), activation='relu', padding='valid', name='penultimate_convs'+str(i))(h_stack_out)

        res = Conv2D(self.nb_channels, (1, 1), padding='valid')(h_stack_out)
        return res


    def build_model(self,input_img=None):
        ''' build conditional PixelCNN model '''
        input_img = Input(shape=(self.input_size[0], self.input_size[1], self.nb_channels), name='input_image')

        if self.conditional:
            latent_vector = Input(shape=(self.latent_dim,), name='latent_vector')
            self.predicted = self._build_layers(input_img, latent_vector)
            self.model = Model([input_img, latent_vector], self.predicted)
        else:
            self.predicted = self._build_layers(input_img)
            self.model = Model(input_img, self.predicted)

        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #self.run_metadata = tf.RunMetadata()

        #self.model.compile(optimizer=self.optimizer, loss=self.loss, options=run_options, run_metadata=self.run_metadata)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
    
    def blockcnn_loss(self):
        return K.sum(K.square(self.input_img-self.predicted),axis=-1)

    def fit(
        self,
        x,
        y,
        batch_size,
        nb_epoch,
        validation_data=None,
        shuffle=True):
        ''' call fit function
        Args:
            x (np.ndarray or [np.ndarray, np.ndarray])  : Input data for training
            y (np.ndarray)                              : Label data for training 
            samples_per_epoch (int)                     : Number of data for each epoch
            nb_epoch (int)                              : Number of epoches
            validation_data ((np.ndarray, np.ndarray))  : Validation data
            nb_val_samples (int)                        : Number of data yielded by validation generator
            shuffle (bool)                              : if True, shuffled randomly
        '''
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer],  # TODO-A: Deactivate early stopping - self.earlystopping],
            validation_data=validation_data,
            shuffle=shuffle
        )
        #trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
        #with open('timeline.json', 'w') as f:
            #f.write(trace.generate_chrome_trace_format())

    def fit_generator(
        self,
        train_generator,
        samples_per_epoch,
        nb_epoch,
        validation_data=None,
        nb_val_samples=10000):
        ''' call fit_generator function
        Args:
            train_generator (object)        : image generator built by "build_generator" function
            samples_per_epoch (int)         : Number of data for each epoch
            nb_epoch (int)                  : Number of epoches
            validation_data (object/array)  : generator object or numpy.ndarray
            nb_val_samples (int)            : Number of data yielded by validation generator
        '''
        self.model.fit_generator(
            generator=train_generator,
            samples_per_epoch=samples_per_epoch,
            epochs=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer], # TODO-A: Deactivate early stopping - self.earlystopping],
            validation_data=validation_data,
            nb_val_samples=nb_val_samples
        )


    def load_model(self, checkpoint_file):
        ''' restore model from checkpoint file (.hdf5) '''
        self.model = load_model(checkpoint_file)

    def export_to_json(self, save_root):
        ''' export model architecture config to json file '''
        with open(os.path.join(save_root, 'pixelcnn_model.json'), 'w') as f:
            f.write(self.model.to_json())

    def export_to_yaml(self, save_root):
        ''' export model architecture config to yaml file '''
        with open(os.path.join(save_root, 'pixelcnn_model.yml'), 'w') as f:
            f.write(self.model.to_yaml())


    @classmethod
    def predict(self, x, batch_size):
        ''' generate image pixel by pixel
        Args:
            x or [x,h] (x,h: numpy.ndarray : x = input image, h = latent vector
        Returns:
            predict (numpy.ndarray)        : generated image
        '''
        return self.model.predict(x, batch_size)


    def print_train_parameters(self, save_root):
        ''' print parameter list file '''
        print('\n########## PixelCNN options ##########')
        print('input_size\t: %s' % (self.input_size,))
        print('nb_pixelcnn_layers: %s' % self.nb_pixelcnn_layers)
        print('nb_filters\t: %s' % self.nb_filters)
        print('filter_size_1st\t: %s' % (self.filter_size_1st,))
        print('filter_size\t: %s' % (self.filter_size,))
        print('conditional\t: %s' % self.conditional)
        print('nb_channels\t: %s' % self.nb_channels)
        print('optimizer\t: %s' % self.optimizer)
        #print('loss\t\t: %s' % self.loss)
        print('es_patience\t: %s' % self.es_patience)
        print('save_root\t: %s' % save_root)
        print('save_best_only\t: %s' % self.save_best_only)
        print('\n')

    def export_train_parameters(self, save_root):
        ''' export parameter list file '''
        with open(os.path.join(save_root, 'parameters.txt'), 'w') as txt_file:
            txt_file.write('########## PixelCNN options ##########\n')
            txt_file.write('input_size\t: %s\n' % (self.input_size,))
            txt_file.write('nb_pixelcnn_layers: %s\n' % self.nb_pixelcnn_layers)
            txt_file.write('nb_filters\t: %s\n' % self.nb_filters)
            txt_file.write('filter_size_1st\t: %s\n' % (self.filter_size_1st,))
            txt_file.write('filter_size\t: %s\n' % (self.filter_size,))
            txt_file.write('conditional\t: %s\n' % self.conditional)
            txt_file.write('nb_channels\t: %s\n' % self.nb_channels)
            txt_file.write('optimizer\t: %s\n' % self.optimizer)
            #txt_file.write('loss\t\t: %s\n' % self.loss)
            txt_file.write('es_patience\t: %s\n' % self.es_patience)
            txt_file.write('save_root\t: %s\n' % save_root)
            txt_file.write('save_best_only\t: %s\n' % self.save_best_only)
            txt_file.write('\n')
