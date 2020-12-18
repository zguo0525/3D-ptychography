#!/usr/bin/env python
# coding: utf-8

# In[11]:


from __future__ import print_function, division, absolute_import

from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.layers import Layer, Flatten, Input, Lambda, Reshape
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras import optimizers, regularizers

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback, CSVLogger, ReduceLROnPlateau

from tensorflow.keras import backend as K

import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 
import h5py as hp
import math 
import argparse


def weighted_layer_npcc(y_pred, y_true):
    weights_Nz = tf.constant([0.016 * 0.6, 0.03 * 0.7, 0.04 * 4, 0.54 * 4, 0.027 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.063 * 2,
                              0.07 * 2, 0.063 * 2, 0.08 * 2, 0.063 * 2, 0.08 * 2, 0.068 * 2, 0.08 * 2, 0.8 * 2, 0.85 * 2, 0.8 * 2, 0.85 * 2])
    weights_Nz **= -1
    weights_Nz /= K.sum(weights_Nz) # Nz
    
    nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
    nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
    nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz
    
    den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
    den_true = K.std(y_true, axis=(1,2))
    den = K.clip(den_pred * den_true, K.epsilon(), None)
    
    npcc_loss = (-1) * K.sum(nom / den * weights_Nz, axis=-1)
    
    return npcc_loss


def layer_npcc(y_pred, y_true):
    nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
    nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
    nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz
    
    den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
    den_true = K.std(y_true, axis=(1,2))
    den = K.clip(den_pred * den_true, K.epsilon(), None)
    
    npcc_loss = (-1) * K.mean(nom / den, axis=-1)
    
    return npcc_loss


def weighted_layer_mean_abs_error(y_pred, y_true):
    weights_Nz = tf.constant([0.016 * 0.6, 0.03 * 0.7, 0.04 * 4, 0.54 * 4, 0.027 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.063 * 2,
                              0.07 * 2, 0.063 * 2, 0.08 * 2, 0.063 * 2, 0.08 * 2, 0.068 * 2, 0.08 * 2, 0.8 * 2, 0.85 * 2, 0.8 * 2, 0.85 * 2])
    weights_Nz **= -1
    weights_Nz /= K.sum(weights_Nz) # Nz
    
    abs_error = K.abs(y_pred - y_true) # B x Ny x Nx x Nz
    mean_abs_error = K.mean(abs_error, axis=(1,2)) # B x Nz
    weighted_layer_mean_abs_error = K.sum(mean_abs_error * weights_Nz, axis=-1) # B
    
    return weighted_layer_mean_abs_error


def shape_list(x):
    static = x.get_shape().as_list()
    shape = K.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)

    return ret


def split_heads_2d(inputs, Nh):
    B, H, W, d = shape_list(inputs)
    ret_shape = [B, H, W, Nh, d//Nh]
    split = K.reshape(inputs, ret_shape)

    return K.permute_dimensions(split, (0,3,1,2,4))


def combine_heads_2d(inputs):
    transposed = K.permute_dimensions(inputs, (0,2,3,1,4))
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]

    return K.reshape(transposed, ret_shape)


class MultiHeadAttention_SingleAxis(Layer):
    def __init__(self, d_model, filters_in, num_heads, attention_axis):
        super(MultiHeadAttention_SingleAxis, self).__init__()
        
        self.d_model = d_model
        self.filters_in = filters_in
        self.Nh = num_heads
        self.attention_axis = attention_axis
        self.dropout_rate = 0.1

        self.Wq = Dense(self.d_model)
        self.Wk = Dense(self.d_model)
        self.Wv = Dense(self.d_model)
        self.Wo = Dense(self.filters_in)
        
    def call(self, x):
        dq = self.d_model
        dk = self.d_model
        dv = self.d_model
        
        dqh = dq // self.Nh
        dkh = dk // self.Nh
        dvh = dv // self.Nh

        q = split_heads_2d(self.Wq(x), self.Nh) # B x Nh x H x W x dqh
        q *= dqh ** -0.5
        k = split_heads_2d(self.Wk(x), self.Nh)
        v = split_heads_2d(self.Wv(x), self.Nh)
        
        if self.attention_axis == 1: # H
            q = K.permute_dimensions(q, (0,1,3,2,4)) # B x Nh x W x H x dqh
            k = K.permute_dimensions(k, (0,1,3,2,4))
            v = K.permute_dimensions(v, (0,1,3,2,4))
            
        attn_compat = tf.matmul(q, k, transpose_b = True)
        attn_prob = K.softmax(attn_compat)
        attn_out = tf.matmul(attn_prob, v)
        attn_out = combine_heads_2d(attn_out)
        attn_out = Dropout(self.dropout_rate)(self.Wo(attn_out))
        
        if self.attention_axis == 1:
            attn_out = K.permute_dimensions(attn_out, (0,2,1,3))
            
        return attn_out
    
    
class AttentionAugmentedConv2D(Layer):
    def __init__(self, Fin, kernel_size, strides, dilation_rate):
        super(AttentionAugmentedConv2D, self).__init__()
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.Fin = Fin
        self.d_model = Fin//2
        self.num_heads = 8
        
        self.attn_H = MultiHeadAttention_SingleAxis(self.d_model, self.d_model, self.num_heads, 1)
        self.attn_W = MultiHeadAttention_SingleAxis(self.d_model, self.d_model, self.num_heads, 2)
        
        self.conv = Conv2D(self.Fin - self.d_model, self.kernel_size, strides = self.strides, dilation_rate = self.dilation_rate, 
                           padding='same', kernel_regularizer=regularizers.l2(1e-4))
        
    def call(self, x):
        attn_out_H = self.attn_H(x)
        attn_out_W = self.attn_W(x)
        attn_out = Add()([attn_out_H, attn_out_W])
        conv_out = self.conv(x)
        attn_conv_out = Concatenate()([attn_out, conv_out])
        
        return attn_conv_out
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Fin': self.Fin,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate
        })
        return config
    
    
class AttentionAugmentedConv2DTranspose(Layer):
    def __init__(self, Fin, kernel_size, strides):
        super(AttentionAugmentedConv2DTranspose, self).__init__()
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.Fin = Fin
        self.d_model = Fin//2
        self.num_heads = 8
        
        self.attn_H = MultiHeadAttention_SingleAxis(self.d_model, self.num_heads, 1)
        self.attn_W = MultiHeadAttention_SingleAxis(self.d_model, self.num_heads, 2)
        
        self.conv = Conv2DTranspose(self.Fin - self.d_model, self.kernel_size, self.strides, 
                                    padding='same', kernel_regularizer=regularizers.l2(1e-4))
        
    def call(self, x):
        attn_out_H = self.attn_H(x)
        attn_out_W = self.attn_W(x)
        attn_out = Add()([attn_out_H, attn_out_W])
        conv_out = self.conv(x)
        attn_conv_out = Concatenate()([attn_out, conv_out])
        
        return attn_conv_out
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Fin': self.Fin,
            'kernel_size': self.kernel_size,
            'strides': self.strides
        })
        return config
    
# Model definition
# 1. Encoder from PhENN - we will use measurements as inputs
filter_size = [48, 96, 192, 384, 512, 512]
deconv_filter_size = 48
num_rows = 128
num_cols = 128
num_layers = 22

# Dimension: (num_angles, num_rows, num_cols, num_channels) excluding num_examples
basic_3d_res_in = Input(shape = (num_rows, num_cols, num_layers))

# DRB1
d1_bn = BatchNormalization()(basic_3d_res_in)
d1_relu = Activation('relu')(d1_bn)
d1_c = Conv2D(filter_size[0], (3, 3), strides=(2, 2), padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d1_relu)
d1_c_bn = BatchNormalization()(d1_c)
d1_c_bn_relu = Activation('relu')(d1_c_bn)
d1_01 = Conv2D(filter_size[0], (3, 3),strides=(1, 1),padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(d1_c_bn_relu)
d1_02 = Conv2D(filter_size[0],(1, 1),strides=(2, 2),padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(basic_3d_res_in)
d1_0 = Add()([d1_01, d1_02])

d1_0 = BatchNormalization()(d1_0)
dr1_relu = Activation('relu')(d1_0)
dr1_c = AttentionAugmentedConv2D(filter_size[0], (3, 3), strides=(1, 1), dilation_rate=(1, 1))(dr1_relu)
dr1_c_bn = BatchNormalization()(dr1_c)
dr1_c_bn_relu = Activation('relu')(dr1_c_bn)
dr1_c_out = AttentionAugmentedConv2D(filter_size[0], (3, 3), strides=(1, 1), dilation_rate=(1, 1))(dr1_c_bn_relu)
d1_out = Add()([dr1_c_out, d1_0])
d1_out = Dropout(2e-2)(d1_out)


# DRB2 ~ DRB4
d2_bn = BatchNormalization()(d1_out)
d2_relu = Activation('relu')(d2_bn)
d2_c = Conv2D(filter_size[1], (3, 3), strides=(2, 2), padding='same', 
             kernel_regularizer=regularizers.l2(1e-4))(d2_relu)
d2_c_bn = BatchNormalization()(d2_c)
d2_c_bn_relu = Activation('relu')(d2_c_bn)
d2_01 = Conv2D(filter_size[1], (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d2_c_bn_relu)
d2_02 = Conv2D(filter_size[1], (1, 1), strides=(2, 2),padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d1_out)
d2_0 = Add()([d2_01, d2_02])

dr2_bn = BatchNormalization()(d2_0)
dr2_relu = Activation('relu')(dr2_bn)
dr2_c = AttentionAugmentedConv2D(filter_size[1],(3, 3),strides=(1, 1), dilation_rate=(2, 2))(dr2_relu)
dr2_c_bn = BatchNormalization()(dr2_c)
dr2_c_bn_relu = Activation('relu')(dr2_c_bn)
dr2_c_out = AttentionAugmentedConv2D(filter_size[1],(3, 3), strides=(1, 1), dilation_rate=(2, 2))(dr2_c_bn_relu)
d2_out = Add()([dr2_c_out, d2_0])
d2_out = Dropout(2e-2)(d2_out)


d3_bn = BatchNormalization()(d2_out)
d3_relu = Activation('relu')(d3_bn)
d3_c = Conv2D(filter_size[2], (3, 3), strides=(2, 2), padding='same', 
             kernel_regularizer=regularizers.l2(1e-4))(d3_relu)
d3_c_bn = BatchNormalization()(d3_c)
d3_c_bn_relu = Activation('relu')(d3_c_bn)
d3_01 = Conv2D(filter_size[2], (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d3_c_bn_relu)
d3_02 = Conv2D(filter_size[2], (1, 1), strides=(2, 2),padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d2_out)
d3_0 = Add()([d3_01,d3_02])

dr3_bn = BatchNormalization()(d3_0)
dr3_relu = Activation('relu')(dr3_bn)
dr3_c = AttentionAugmentedConv2D(filter_size[2], (3, 3), strides=(1, 1), dilation_rate=(2, 2))(dr3_relu)
dr3_c_bn = BatchNormalization()(dr3_c)
dr3_c_bn_relu = Activation('relu')(dr3_c_bn)
dr3_c_out = AttentionAugmentedConv2D(filter_size[2], (3, 3), strides=(1, 1), dilation_rate=(2, 2))(dr3_c_bn_relu)
d3_out = Add()([dr3_c_out, d3_0])
d3_out = Dropout(2e-2)(d3_out)


d4_bn = BatchNormalization()(d3_out)
d4_relu = Activation('relu')(d4_bn)
d4_c = Conv2D(filter_size[3], (3, 3), strides=(2, 2), padding='same', 
             kernel_regularizer=regularizers.l2(1e-4))(d4_relu)
d4_c_bn = BatchNormalization()(d4_c)
d4_c_bn_relu = Activation('relu')(d4_c_bn)
d4_01 = Conv2D(filter_size[3], (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d4_c_bn_relu)
d4_02 = Conv2D(filter_size[3], (1, 1), strides=(2, 2),padding='same', 
              kernel_regularizer=regularizers.l2(1e-4))(d3_out)
d4_0 = Add()([d4_01,d4_02])

dr4_bn = BatchNormalization()(d4_0)
dr4_relu = Activation('relu')(dr4_bn)
dr4_c = AttentionAugmentedConv2D(filter_size[3], (3, 3), strides=(1, 1), dilation_rate=(2, 2))(dr4_relu)
dr4_c_bn = BatchNormalization()(dr4_c)
dr4_c_bn_relu = Activation('relu')(dr4_c_bn)
dr4_c_out = AttentionAugmentedConv2D(filter_size[3], (3, 3), strides=(1, 1), dilation_rate=(2, 2))(dr4_c_bn_relu)
d4_out = Add()([dr4_c_out, d4_0])
d4_out = Dropout(2e-2)(d4_out)

# URB1
u1_bn = BatchNormalization()(d4_out)
u1_relu = Activation('relu')(u1_bn)
u1_ct = Conv2DTranspose(filter_size[4], (3, 3), strides=(2, 2), padding='same', 
                        kernel_regularizer=regularizers.l2(1e-4))(u1_relu)
u1_ct_bn = BatchNormalization()(u1_ct)
u1_ct_relu = Activation('relu')(u1_ct_bn)
u1_01 = Conv2D(filter_size[4], (3, 3), strides=(1, 1), padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(u1_ct_relu)
u1_02 = Conv2DTranspose(filter_size[4], (2, 2), strides=(2, 2), padding='same',
                        kernel_regularizer=regularizers.l2(1e-4))(d4_out)
u1_0 = Add()([u1_01, u1_02])

ur1_bn = BatchNormalization()(u1_0)
ur1_relu = Activation('relu')(ur1_bn)
ur1_c = AttentionAugmentedConv2D(filter_size[4], (3, 3), strides=(1, 1), dilation_rate=(1, 1))(ur1_relu)
ur1_c_bn = BatchNormalization()(ur1_c)
ur1_c_bn_relu = Activation('relu')(ur1_c_bn)
ur1_c_out = AttentionAugmentedConv2D(filter_size[4], (3, 3), strides=(1, 1), dilation_rate=(1, 1))(ur1_c_bn_relu)

u1_1 = Add()([ur1_c_out, u1_0])
u1_out = Concatenate()([u1_1, d3_out])
u1_out = Dropout(2e-2)(u1_out)


# URB2
u2_bn = BatchNormalization()(u1_out)
u2_relu = Activation('relu')(u2_bn)
u2_ct = Conv2DTranspose(filter_size[3], (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularizers.l2(1e-4))(u2_relu)
u2_ct_bn = BatchNormalization()(u2_ct)
u2_ct_relu = Activation('relu')(u2_ct_bn)
u2_01 = Conv2D(filter_size[3], (3, 3), strides=(1, 1), padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(u2_ct_relu)
u2_02 = Conv2DTranspose(filter_size[3], (2, 2), strides=(2, 2),padding='same',
                        kernel_regularizer=regularizers.l2(1e-4))(u1_out)
u2_0 = Add()([u2_01, u2_02])

ur2_bn = BatchNormalization()(u2_0)
ur2_relu = Activation('relu')(ur2_bn)
ur2_c = AttentionAugmentedConv2D(filter_size[3], (3, 3), strides=(1, 1), dilation_rate=(1, 1))(ur2_relu)
ur2_c_bn = BatchNormalization()(ur2_c)
ur2_c_bn_relu = Activation('relu')(ur2_c_bn)
ur2_c_out = AttentionAugmentedConv2D(filter_size[3], (3, 3), strides=(1, 1), dilation_rate=(1, 1))(ur2_c_bn_relu)

u2_1 = Add()([ur2_c_out, u2_0])
u2_out = Concatenate()([u2_1, d2_out])
u2_out = Dropout(2e-2)(u2_out)


# URB3
u3_bn = BatchNormalization()(u2_out)
u3_relu = Activation('relu')(u3_bn)
u3_ct = Conv2DTranspose(filter_size[2], (3, 3), strides=(2, 2), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(u3_relu)
u3_ct_bn = BatchNormalization()(u3_ct)
u3_ct_relu = Activation('relu')(u3_ct_bn)
u3_01 = Conv2D(filter_size[2], (3,3),strides=(1,1),padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(u3_ct_relu)
u3_02 = Conv2DTranspose(filter_size[2],(2,2),strides=(2,2),padding='same',
                        kernel_regularizer=regularizers.l2(1e-4))(u2_out)
u3_0 = Add()([u3_01, u3_02])

ur3_bn = BatchNormalization()(u3_0)
ur3_relu = Activation('relu')(ur3_bn)
ur3_c = AttentionAugmentedConv2D(filter_size[2], (3,3), strides=(1,1), dilation_rate=(1, 1))(ur3_relu)
ur3_c_bn = BatchNormalization()(ur3_c)
ur3_c_bn_relu = Activation('relu')(ur3_c_bn)
ur3_c_out = AttentionAugmentedConv2D(filter_size[2], (3,3), strides=(1,1), dilation_rate=(1, 1))(ur3_c_bn_relu)

u3_1 = Add()([ur3_c_out,u3_0])
u3_out = Concatenate()([u3_1,d1_out])
u3_out = Dropout(2e-2)(u3_out)


# URB4
u4_bn = BatchNormalization()(u3_out)
u4_relu = Activation('relu')(u4_bn)
u4_ct = Conv2DTranspose(filter_size[1], (3, 3), strides=(2, 2), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(u4_relu)
u4_ct_bn = BatchNormalization()(u4_ct)
u4_ct_relu = Activation('relu')(u4_ct_bn)
u4_01 = Conv2D(filter_size[1], (3,3),strides=(1,1),padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(u4_ct_relu)
u4_02 = Conv2DTranspose(filter_size[1],(2,2),strides=(2,2),padding='same',
                        kernel_regularizer=regularizers.l2(1e-4))(u3_out)
u4_0 = Add()([u4_01, u4_02])

ur4_bn = BatchNormalization()(u4_0)
ur4_relu = Activation('relu')(ur4_bn)
ur4_c = AttentionAugmentedConv2D(filter_size[1],(3,3),strides=(1,1), dilation_rate=(1, 1))(ur4_relu)
ur4_c_bn = BatchNormalization()(ur4_c)
ur4_c_bn_relu = Activation('relu')(ur4_c_bn)
ur4_c_out = AttentionAugmentedConv2D(filter_size[1],(3,3),strides=(1,1), dilation_rate=(1, 1))(ur4_c_bn_relu)

u4_1 = Add()([ur4_c_out,u4_0])
u4_out = Concatenate()([u4_1,basic_3d_res_in])
u4_out = Dropout(2e-2)(u4_out)


# RB1
r1_bn = BatchNormalization()(u4_out)
r1_relu = Activation('relu')(r1_bn)
r1_c = Conv2D(deconv_filter_size,(3,3),strides=(1,1),padding='same',
              kernel_regularizer=regularizers.l2(1e-4))(r1_relu)
r1_c_bn = BatchNormalization()(r1_c)
r1_c_bn_relu = Activation('relu')(r1_c_bn)
r1_c_out_1 = Conv2D(deconv_filter_size,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r1_c_bn_relu)

r1_c_out_2 = Conv2D(deconv_filter_size,(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(u4_out)
r1_out_1 = Add()([r1_c_out_1,r1_c_out_2])

r1_bn_2 = BatchNormalization()(r1_out_1)
r1_relu_2 = Activation('relu')(r1_bn_2)
r1_c_2 = Conv2D(deconv_filter_size,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r1_relu_2)
r1_c_bn_2 = BatchNormalization()(r1_c_2)
r1_c_bn_relu_2 = Activation('relu')(r1_c_bn_2)
r1_c_out_1_2 = Conv2D(deconv_filter_size,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r1_c_bn_relu_2)
r1_c_out_2_2 = Conv2D(deconv_filter_size,(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r1_out_1)
r1_out_2 = Add()([r1_c_out_1_2,r1_c_out_2_2])

r1_out = Dropout(2e-2)(r1_out_2)


# RB2
r2_bn = BatchNormalization()(r1_out)
r2_relu = Activation('relu')(r2_bn)
r2_c = Conv2D(num_layers,(3,3),strides=(1,1),padding='same',
              kernel_regularizer=regularizers.l2(1e-4))(r2_relu)
r2_c_bn = BatchNormalization()(r2_c)
r2_c_bn_relu = Activation('relu')(r2_c_bn)
r2_c_out_1 = Conv2D(num_layers,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r2_c_bn_relu)

r2_c_out_2 = Conv2D(num_layers,(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r1_out)
r2_out_1 = Add()([r2_c_out_1,r2_c_out_2])

r2_bn_2 = BatchNormalization()(r2_out_1)
r2_relu_2 = Activation('relu')(r2_bn_2)
r2_c_2 = Conv2D(num_layers,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r2_relu_2)
r2_c_bn_2 = BatchNormalization()(r2_c_2)
r2_c_bn_relu_2 = Activation('relu')(r2_c_bn_2)
r2_c_out_1_2 = Conv2D(num_layers,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r2_c_bn_relu_2)
r2_c_out_2_2 = Conv2D(num_layers,(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))(r2_out_1)
r2_out_2 = Add()([r2_c_out_1_2,r2_c_out_2_2])

r2_out = Dropout(2e-2)(r2_out_2)

basic_3d_res = Model(inputs = basic_3d_res_in, outputs = r2_out)
basic_3d_res.summary()


# In[ ]:


def g_loss_npcc(generated_image, true_image):
    fsp=generated_image-K.mean(generated_image,axis=(1,2,3),keepdims=True)
    fst=true_image-K.mean(true_image,axis=(1,2,3),keepdims=True)
    
    devP=K.std(generated_image,axis=(1,2,3))
    devT=K.std(true_image,axis=(1,2,3))
    
    npcc_loss=(-1)*K.mean(fsp*fst,axis=(1,2,3))/K.clip(devP*devT,K.epsilon(),None)    ## (BL,1)
    return npcc_loss


def weighted_layer_npcc(y_pred, y_true):
    weights_Nz = tf.constant([0.016 * 0.6, 0.03 * 0.7, 0.04 * 4, 0.54 * 4, 0.027 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.063 * 2,
                              0.07 * 2, 0.063 * 2, 0.08 * 2, 0.063 * 2, 0.08 * 2, 0.068 * 2, 0.08 * 2, 0.8 * 2, 0.85 * 2, 0.8 * 2, 0.85 * 2])
    weights_Nz **= -1
    weights_Nz /= K.sum(weights_Nz) # Nz
    
    nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
    nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
    nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz
    
    den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
    den_true = K.std(y_true, axis=(1,2))
    den = K.clip(den_pred * den_true, K.epsilon(), None)
    
    npcc_loss = (-1) * K.sum(nom / den * weights_Nz, axis=-1)
    
    return npcc_loss


def layer_npcc(y_pred, y_true):
    nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
    nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
    nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz
    
    den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
    den_true = K.std(y_true, axis=(1,2))
    den = K.clip(den_pred * den_true, K.epsilon(), None)
    
    npcc_loss = (-1) * K.mean(nom / den, axis=-1)
    
    return npcc_loss


optadam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
optsgd = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
basic_3d_res.compile(optimizer=optadam,loss=layer_npcc, metrics=[layer_npcc])
# basic_3d_res.compile(optimizer=optadam,loss=layer_npcc, metrics=[layer_npcc])

# In[ ]:


# loading training data
'''
wavelength = 0.14e-9
n0 = 1.0
n1 = 1.0
k = 2 * np.pi / wavelength
k0 = k

Nz = 22
Ny = 128
Nx = Ny
num_training = 2000
training_input = np.zeros(shape=(num_training, Nx, Ny, Nz)).astype(complex)
training_output = np.zeros(shape=(num_training, Nx, Ny, Nz))

layer_thickness = np.asarray([0.016, 0.03, 0.04, 0.054, 0.027, 0.07, 0.06, 0.07, 0.06, 0.07, 0.063, 0.07, 0.063, 0.08, 0.063, 0.08, 0.068, 0.08, 0.8, 0.85, 0.8, 0.85]) * 1e-6

dn = np.zeros((512,512,22))
dn[:, :, 1] = -6e-6
dn[:, :, 2] = -7e-6
dn[:, :, 3:4] = -4e-5
dn[:, :, 5:22] = -2e-5

import h5py
from tqdm import tqdm

for i in tqdm(range(1, num_training + 1)):
    filename = 'layers/layers_' + str(i) + '.mat'
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        object = list(f[a_group_key])
        object = np.asarray(object)
        object = np.transpose(object)
    training_output[i-1, :, :, :] = object[192:320, 192:320, :]
    
for i in tqdm(range(1, num_training + 1)):
    result = np.load('approximates/result' +str(i) + '.npy')
    result = result[:, :, :, 0] + 1j * result[:, :, :, 1]
    training_input[i-1, :, :, :] = result[192:320, 192:320, :]
    
validation_input = training_input[num_training - 100 : num_training - 50]
test_input = training_input[num_training - 50 : num_training]
training_input = training_input[:num_training - 100]

validation_output = training_output[num_training - 100 : num_training - 50]
test_output = training_output[num_training - 50 : num_training]
training_output = training_output[:num_training - 100]

print(training_input.shape)
print(validation_input.shape)
print(test_input.shape)
'''




class LearningRateBasedStopping(tf.keras.callbacks.Callback):
    def __init__(self, limit_lr):
        super(LearningRateBasedStopping, self).__init__()
        self.limit_lr = limit_lr
	
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, lr))
                                    
        if lr < self.limit_lr:
            self.model.stop_training = True


def scheduler(epoch, lr):
    warmup_epochs = 20
    lr = 512 ** -1.2 * np.min([(epoch + 1) ** -0.5, (epoch + 1) * warmup_epochs ** -1.5])

    return lr

# In[ ]:


batch_size = 5
num_epochs = 200

weight_folder = 'weights/attn_conv_2d_v2/'
if not os.path.exists(weight_folder):
    os.makedirs(weight_folder)

csv_logger = CSVLogger('log/attn_conv_2d_v2.log')
checkpoint = ModelCheckpoint(filepath = weight_folder + '{epoch:02d}.hdf5',
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
lr_stopping = LearningRateBasedStopping(1e-8)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, verbose=1, mode='auto', min_delta=0.0001,
                             cooldown=0, min_lr=1e-8)
lr_scheduler = LearningRateScheduler(scheduler, verbose = 1)
callbacks_list = [checkpoint, csv_logger, lr_stopping, reducelr]
basic_3d_res.load_weights('weights/attn_conv_2d_v2/37.hdf5')

experiment = np.load('experiment_approx_layer.npy')
experiment = experiment[:, :, :, 0] + 1j * experiment[:, :, :, 1]
experiment = np.expand_dims(experiment[192:320, 192:320, :], axis=0)
print(np.shape(experiment))



rec = basic_3d_res.predict(experiment, batch_size = 1, verbose =1)
#basic_3d_res.fit(training_input, training_output, batch_size = batch_size, epochs = num_epochs,
#               validation_data = (validation_input, validation_output), verbose = 1, callbacks = callbacks_list)

np.save('rec_attn_cov_layer', rec)

