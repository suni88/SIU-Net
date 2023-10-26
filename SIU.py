import os 
import numpy as np 
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add 
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D 
from keras.models import Model, model_from_json 
from keras.optimizers import * 
# from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model 
from keras import backend as K  
from data import *


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None): 

    ''' 

    2D Convolutional layers 
    Arguments: 
        x {keras layer} -- input layer  
        filters {int} -- number of filters 
        num_row {int} -- number of rows in filters 
        num_col {int} -- number of columns in filters 
        
    Keyword Arguments: 
        padding {str} -- mode of padding (default: {'same'}) 
        strides {tuple} -- stride of convolution operation (default: {(1, 1)}) 
        activation {str} -- activation function (default: {'relu'}) 
        name {str} -- name of the layer (default: {None}) 

    Returns: 
        [keras layer] -- [output layer] 
    ''' 

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x) 
    x = BatchNormalization(axis=3, scale=False)(x) 

    if(activation == None): 
        return x 

    x = Activation(activation, name=name)(x) 
    
    return x 

def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None): 

    ''' 
    2D Transposed Convolutional layers 

    Arguments: 
        x {keras layer} -- input layer  
        filters {int} -- number of filters 
        num_row {int} -- number of rows in filters 
        num_col {int} -- number of columns in filters 

    Keyword Arguments: 
        padding {str} -- mode of padding (default: {'same'}) 
        strides {tuple} -- stride of convolution operation (default: {(2, 2)}) 
        name {str} -- name of the layer (default: {None}) 

    Returns: 
        [keras layer] -- [output layer] 
    ''' 
    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x) 
    x = BatchNormalization(axis=3, scale=False)(x) 
  
    return x 


def siu_inception(U, inp, alpha = 1.67):
    '''
    Inception Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp

    shortcut = conv2d_bn(shortcut, U, 1, 1, activation=None, padding='same')

    conv1 = conv2d_bn(inp, U, 1, 1, activation='relu', padding='same')    
    conv11 = conv2d_bn(conv1, U, 3, 3, activation='relu', padding='same')

    conv2 = conv2d_bn(inp, U, 1, 1, activation='relu', padding='same')
    conv22 = conv2d_bn(conv2, U, 3, 3,activation='relu', padding='same')

    pool = MaxPooling2D(pool_size=(1, 1))(inp)
    convp = conv2d_bn(pool, U, 3, 3,activation='relu', padding='same')
    
    # out = concatenate([conv11, conv22, convp], axis=3)
    out = conv11+conv22+convp
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)
    
    return out



def ResPath(filters, length, inp): 

    ''' 
    ResPath 

    Arguments: 
        filters {int} -- [description] 
        length {int} -- length of ResPath 
        inp {keras layer} -- input layer  

    Returns: 
        [keras layer] -- [output layer] 
    ''' 

    shortcut = inp 
    shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same') 

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same') 
    out = add([shortcut, out]) 
    out = Activation('relu')(out) 

    out = BatchNormalization(axis=3)(out) 

    for i in range(length-1): 

        shortcut = out 
        shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same') 

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same') 
        out = add([shortcut, out]) 
        out = Activation('relu')(out) 
        out = BatchNormalization(axis=3)(out) 

    return out 

def siu_net(height, width, n_channels): 

    ''' 
    SIU-Net 
    Arguments: 
        height {int} -- height of image  
        width {int} -- width of image  
        n_channels {int} -- number of channels in image 

    Returns: 
        [keras model] -- SIU-Net model 

    ''' 

    inputs = Input((height, width, n_channels)) 

    mresblock1 = siu_inception(32, inputs) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1) 
    mresblock1 = ResPath(32, 4, mresblock1) 

    mresblock2 = siu_inception(32*2, pool1) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2) 
    mresblock2 = ResPath(32*2, 3, mresblock2) 

    mresblock3 = siu_inception(32*4, pool2) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3) 
    mresblock3 = ResPath(32*4, 2, mresblock3) 

    mresblock4 = siu_inception(32*8, pool3) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4) 
    mresblock4 = ResPath(32*8, 1, mresblock4) 

    mresblock5 = siu_inception(32*16, pool4) 
    
    up6 = concatenate([Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = siu_inception(32*8, up6) 
    mresblock6_up1 = Conv2DTranspose(8 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock6) 
    mresblock6_up2 = Conv2DTranspose(8 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock6_up1) 
    mresblock6_up3 = Conv2DTranspose(8 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock6_up2) 
    mresblock6=Conv2DTranspose(2 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock6) 

    up7=concatenate([mresblock3,mresblock6,mresblock6_up1], axis=3) 
    mresblock7 = siu_inception(32*4, up7) 
    mresblock7_up1 = Conv2DTranspose(4 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock7) 
    mresblock7_up2 = Conv2DTranspose(4 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock7_up1) 
    mresblock7_up3 = Conv2DTranspose(4 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock7_up2) 
    mresblock7=Conv2DTranspose(2 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock7) 

    up8=concatenate([mresblock2,mresblock7,mresblock7_up1,mresblock6_up2],axis=3) 
    mresblock8 = siu_inception(32*2, up8) 
    mresblock8_up1 = Conv2DTranspose(2 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock8) 
    mresblock8_up2 = Conv2DTranspose(2 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock8_up1) 
    mresblock8_up3 = Conv2DTranspose(2 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock8_up2) 
    mresblock8=Conv2DTranspose(2 * 32, (2, 2), strides=(2, 2), padding="same")(mresblock8) 

    up9=concatenate([mresblock1,mresblock8,mresblock7_up2,mresblock6_up3,mresblock8_up1],axis=3) 
    mresblock9 = siu_inception(32, up9) 

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid') 

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model 
