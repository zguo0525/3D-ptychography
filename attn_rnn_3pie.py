from __future__ import print_function, division, absolute_import
from functools import partial, update_wrapper
from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.layers import Flatten, Input, Lambda, Reshape, Layer, Multiply, Subtract
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D, ConvLSTM2D, MaxPooling3D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D, LayerNormalization
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


class BiasLayer(Layer):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.reg = 1e-4

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True,
                                    regularizer=regularizers.l2(self.reg))
        
    def call(self, x):
        return x + self.bias
    
    
class FlexibleConv3D(Layer):
    def __init__(self, conv_filter, kernel_size, strides, dilation_rate, reg=1e-4):
        super(FlexibleConv3D, self).__init__()
        
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_filter = conv_filter
        
        # self.kernel_size = (1,3,3), self.strides = (1,1,1) or (1,2,2)
        self.Wxy = Conv3D(conv_filter, (self.kernel_size, self.kernel_size, 1), strides=self.strides, dilation_rate = self.dilation_rate,
                          padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=True, bias_regularizer=regularizers.l2(reg)) 
        self.Wz = Conv3D(conv_filter, (1,1,self.kernel_size), strides=self.strides, dilation_rate = self.dilation_rate,
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=True, bias_regularizer=regularizers.l2(reg))
    
    def call(self, p):
        qxy = self.Wxy(p)
        qz = self.Wz(p)

        out = Add()([qxy, qz])
            
        return out
    

class Encoder(Layer):
    def __init__(self, filter_size):
        super(Encoder, self).__init__()
        
        self.filter_size = filter_size

        self.d1_bn = BatchNormalization()
        self.d1_relu = Activation('relu')
        self.d1_c = FlexibleConv3D(self.filter_size[0], 3, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.d1_c_bn = BatchNormalization()
        self.d1_c_bn_relu = Activation('relu')
        self.d1_01 = FlexibleConv3D(self.filter_size[0], 3, strides=(1, 1, 1), dilation_rate=(1, 1, 1))
        self.d1_02 = FlexibleConv3D(self.filter_size[0], 1, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.d1_0 = BatchNormalization()
        self.dr1_relu = Activation('relu')
        self.dr1_c = FlexibleConv3D(self.filter_size[0], 3, strides=(1, 1, 1), dilation_rate=(1, 1, 1))
        self.dr1_c_bn = BatchNormalization()
        self.dr1_c_bn_relu = Activation('relu')
        self.dr1_c_out = FlexibleConv3D(self.filter_size[0], 3, strides=(1, 1, 1), dilation_rate=(1, 1, 1))

        self.d2_bn = BatchNormalization()
        self.d2_relu = Activation('relu')
        self.d2_c = FlexibleConv3D(self.filter_size[1], 3, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.d2_c_bn = BatchNormalization()
        self.d2_c_bn_relu = Activation('relu')
        self.d2_01 = FlexibleConv3D(self.filter_size[1], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))
        self.d2_02 = FlexibleConv3D(self.filter_size[1], 1, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.dr2_bn = BatchNormalization()
        self.dr2_relu = Activation('relu')
        self.dr2_c = FlexibleConv3D(self.filter_size[1], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))
        self.dr2_c_bn = BatchNormalization()
        self.dr2_c_bn_relu = Activation('relu')
        self.dr2_c_out = FlexibleConv3D(self.filter_size[1], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))

        self.d3_bn = BatchNormalization()
        self.d3_relu = Activation('relu')
        self.d3_c = FlexibleConv3D(self.filter_size[2], 3, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.d3_c_bn = BatchNormalization()
        self.d3_c_bn_relu = Activation('relu')
        self.d3_01 = FlexibleConv3D(self.filter_size[2], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))
        self.d3_02 = FlexibleConv3D(self.filter_size[2], 1, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.dr3_bn = BatchNormalization()
        self.dr3_relu = Activation('relu')
        self.dr3_c = FlexibleConv3D(self.filter_size[2], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))
        self.dr3_c_bn = BatchNormalization()
        self.dr3_c_bn_relu = Activation('relu')
        self.dr3_c_out = FlexibleConv3D(self.filter_size[2], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))

        self.d4_bn = BatchNormalization()
        self.d4_relu = Activation('relu')
        self.d4_c = FlexibleConv3D(self.filter_size[3], 3, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.d4_c_bn = BatchNormalization()
        self.d4_c_bn_relu = Activation('relu')
        self.d4_01 = FlexibleConv3D(self.filter_size[3], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))
        self.d4_02 = FlexibleConv3D(self.filter_size[3], 1, strides=(2, 2, 1), dilation_rate=(1, 1, 1))
        self.dr4_bn = BatchNormalization()
        self.dr4_relu = Activation('relu')
        self.dr4_c = FlexibleConv3D(self.filter_size[3], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))
        self.dr4_c_bn = BatchNormalization()
        self.dr4_c_bn_relu = Activation('relu')
        self.dr4_c_out = FlexibleConv3D(self.filter_size[3], 3, strides=(1, 1, 1), dilation_rate=(2, 2, 1))

    def call(self, x):
        d1_bn = self.d1_bn(x)
        d1_relu = self.d1_relu(d1_bn)
        d1_c = self.d1_c(d1_relu)
        d1_c_bn = self.d1_c_bn(d1_c)
        d1_c_bn_relu = self.d1_c_bn_relu(d1_c_bn)
        d1_01 = self.d1_01(d1_c_bn_relu)
        d1_02 = self.d1_02(x)
        d1_0 = Add()([d1_01, d1_02])
        
        d1_0 = self.d1_0(d1_0)
        dr1_relu = self.dr1_relu(d1_0)
        dr1_c = self.dr1_c(dr1_relu)
        dr1_c_bn = self.dr1_c_bn(dr1_c)
        dr1_c_bn_relu = self.dr1_c_bn_relu(dr1_c_bn)
        dr1_c_out = self.dr1_c_out(dr1_c_bn_relu)
        d1_out = Add()([dr1_c_out, d1_0])
        d1_out = Dropout(2e-2)(d1_out)
        
        
        d2_bn = self.d2_bn(d1_out)
        d2_relu = self.d2_relu(d2_bn)
        d2_c = self.d2_c(d2_relu)
        d2_c_bn = self.d2_c_bn(d2_c)
        d2_c_bn_relu = self.d2_c_bn_relu(d2_c_bn)
        d2_01 = self.d2_01(d2_c_bn_relu)
        d2_02 = self.d2_02(d1_out)
        d2_0 = Add()([d2_01, d2_02])

        dr2_bn = self.dr2_bn(d2_0)
        dr2_relu = self.dr2_relu(dr2_bn)
        dr2_c = self.dr2_c(dr2_relu)
        dr2_c_bn = self.dr2_c_bn(dr2_c)
        dr2_c_bn_relu = self.dr2_c_bn_relu(dr2_c_bn)
        dr2_c_out = self.dr2_c_out(dr2_c_bn_relu)
        d2_out = Add()([dr2_c_out, d2_0])
        d2_out = Dropout(2e-2)(d2_out)


        d3_bn = self.d3_bn(d2_out)
        d3_relu = self.d3_relu(d3_bn)
        d3_c = self.d3_c(d3_relu)
        d3_c_bn = self.d3_c_bn(d3_c)
        d3_c_bn_relu = self.d3_c_bn_relu(d3_c_bn)
        d3_01 = self.d3_01(d3_c_bn_relu)
        d3_02 = self.d3_02(d2_out)
        d3_0 = Add()([d3_01,d3_02])

        dr3_bn = self.dr3_bn(d3_0)
        dr3_relu = self.dr3_relu(dr3_bn)
        dr3_c = self.dr3_c(dr3_relu)
        dr3_c_bn = self.dr3_c_bn(dr3_c)
        dr3_c_bn_relu = self.dr3_c_bn_relu(dr3_c_bn)
        dr3_c_out = self.dr3_c_out(dr3_c_bn_relu)
        d3_out = Add()([dr3_c_out, d3_0])
        d3_out = Dropout(2e-2)(d3_out)


        d4_bn = self.d4_bn(d3_out)
        d4_relu = self.d4_relu(d4_bn)
        d4_c = self.d4_c(d4_relu)
        d4_c_bn = self.d4_c_bn(d4_c)
        d4_c_bn_relu = self.d4_c_bn_relu(d4_c_bn)
        d4_01 = self.d4_01(d4_c_bn_relu)
        d4_02 = self.d4_02(d3_out)
        d4_0 = Add()([d4_01,d4_02])

        dr4_bn = self.dr4_bn(d4_0)
        dr4_relu = self.dr4_relu(dr4_bn)
        dr4_c = self.dr4_c(dr4_relu)
        dr4_c_bn = self.dr4_c_bn(dr4_c)
        dr4_c_bn_relu = self.dr4_c_bn_relu(dr4_c_bn)
        dr4_c_out = self.dr4_c_out(dr4_c_bn_relu)
        d4_out = Add()([dr4_c_out, d4_0])
        d4_out = Dropout(2e-2)(d4_out)

        return d4_out


# In[52]:


class Decoder(Layer):
    def __init__(self, filter_size, num_layers):
        super(Decoder, self).__init__()
        
        self.filter_size = filter_size
        self.num_layers = num_layers

        self.u1_bn = BatchNormalization()
        self.u1_relu = Activation('relu')
        self.u1_ct = Conv2DTranspose(self.filter_size[4], (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u1_ct_bn = BatchNormalization()
        self.u1_ct_relu = Activation('relu')
        self.u1_01 = Conv2D(self.filter_size[4], (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u1_02 = Conv2DTranspose(self.filter_size[4], (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur1_bn = BatchNormalization()
        self.ur1_relu = Activation('relu')
        self.ur1_c = Conv2D(self.filter_size[4], (3, 3),strides=(1, 1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.ur1_c_bn = BatchNormalization()
        self.ur1_c_bn_relu = Activation('relu')
        self.ur1_c_out = Conv2D(self.filter_size[4], (3, 3),strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(1e-4))
        
        self.u2_bn = BatchNormalization()
        self.u2_relu = Activation('relu')
        self.u2_ct = Conv2DTranspose(self.filter_size[3], (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u2_ct_bn = BatchNormalization()
        self.u2_ct_relu = Activation('relu')
        self.u2_01 = Conv2D(self.filter_size[3], (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u2_02 = Conv2DTranspose(self.filter_size[3], (2, 2), strides=(2, 2),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur2_bn = BatchNormalization()
        self.ur2_relu = Activation('relu')
        self.ur2_c = Conv2D(self.filter_size[3], (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur2_c_bn = BatchNormalization()
        self.ur2_c_bn_relu = Activation('relu')
        self.ur2_c_out = Conv2D(self.filter_size[3], (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))

        self.u3_bn = BatchNormalization()
        self.u3_relu = Activation('relu')
        self.u3_ct = Conv2DTranspose(self.filter_size[2], (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u3_ct_bn = BatchNormalization()
        self.u3_ct_relu = Activation('relu')
        self.u3_01 = Conv2D(self.filter_size[2], (3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u3_02 = Conv2DTranspose(self.filter_size[2],(2,2),strides=(2,2),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur3_bn = BatchNormalization()
        self.ur3_relu = Activation('relu')
        self.ur3_c = Conv2D(self.filter_size[2],(3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur3_c_bn = BatchNormalization()
        self.ur3_c_bn_relu = Activation('relu')
        self.ur3_c_out = Conv2D(self.filter_size[2],(3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        
        self.u4_bn = BatchNormalization()
        self.u4_relu = Activation('relu')
        self.u4_ct = Conv2DTranspose(self.filter_size[1], (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u4_ct_bn = BatchNormalization()
        self.u4_ct_relu = Activation('relu')
        self.u4_01 = Conv2D(self.filter_size[1], (3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.u4_02 = Conv2DTranspose(self.filter_size[1],(2,2),strides=(2,2),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur4_bn = BatchNormalization()
        self.ur4_relu = Activation('relu')
        self.ur4_c = Conv2D(self.filter_size[1],(3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.ur4_c_bn = BatchNormalization()
        self.ur4_c_bn_relu = Activation('relu')
        self.ur4_c_out = Conv2D(self.filter_size[1],(3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        
        self.r1_bn = BatchNormalization()
        self.r1_relu = Activation('relu')
        self.r1_c = Conv2D(self.filter_size[0],(3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.r1_c_bn = BatchNormalization()
        self.r1_c_bn_relu = Activation('relu')
        self.r1_c_out_1 = Conv2D(self.filter_size[0],(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r1_c_out_2 = Conv2D(self.filter_size[0],(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r1_bn_2 = BatchNormalization()
        self.r1_relu_2 = Activation('relu')
        self.r1_c_2 = Conv2D(self.filter_size[0],(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r1_c_bn_2 = BatchNormalization()
        self.r1_c_bn_relu_2 = Activation('relu')
        self.r1_c_out_1_2 = Conv2D(self.filter_size[0],(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r1_c_out_2_2 = Conv2D(self.filter_size[0],(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))

        self.r2_bn = BatchNormalization()
        self.r2_relu = Activation('relu')
        self.r2_c = Conv2D(self.num_layers,(3,3),strides=(1,1),padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.r2_c_bn = BatchNormalization()
        self.r2_c_bn_relu = Activation('relu')
        self.r2_c_out_1 = Conv2D(self.num_layers,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r2_c_out_2 = Conv2D(self.num_layers,(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r2_bn_2 = BatchNormalization()
        self.r2_relu_2 = Activation('relu')
        self.r2_c_2 = Conv2D(self.num_layers,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r2_c_bn_2 = BatchNormalization()
        self.r2_c_bn_relu_2 = Activation('relu')
        self.r2_c_out_1_2 = Conv2D(self.num_layers,(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))
        self.r2_c_out_2_2 = Conv2D(self.num_layers,(1,1),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-4))

    def call(self, x): 
        u1_bn = self.u1_bn(x)
        u1_relu = self.u1_relu(u1_bn)
        u1_ct = self.u1_ct(u1_relu)
        u1_ct_bn = self.u1_ct_bn(u1_ct)
        u1_ct_relu = self.u1_ct_relu(u1_ct_bn)
        u1_01 = self.u1_01(u1_ct_relu)
        u1_02 = self.u1_02(x)
        u1_0 = Add()([u1_01, u1_02])

        ur1_bn = self.ur1_bn(u1_0)
        ur1_relu = self.ur1_relu(ur1_bn)
        ur1_c = self.ur1_c(ur1_relu)
        ur1_c_bn = self.ur1_c_bn(ur1_c)
        ur1_c_bn_relu = self.ur1_c_bn_relu(ur1_c_bn)
        ur1_c_out = self.ur1_c_out(ur1_c_bn_relu)

        u1_1 = Add()([ur1_c_out, u1_0])
#         u1_out = Concatenate()([u1_1, d3_out])
        u1_out = Dropout(2e-2)(u1_1)


        u2_bn = self.u2_bn(u1_out)
        u2_relu = self.u2_relu(u2_bn)
        u2_ct = self.u2_ct(u2_relu)
        u2_ct_bn = self.u2_ct_bn(u2_ct)
        u2_ct_relu = self.u2_ct_relu(u2_ct_bn)
        u2_01 = self.u2_01(u2_ct_relu)
        u2_02 = self.u2_02(u1_out)
        u2_0 = Add()([u2_01, u2_02])

        ur2_bn = self.ur2_bn(u2_0)
        ur2_relu = self.ur2_relu(ur2_bn)
        ur2_c = self.ur2_c(ur2_relu)
        ur2_c_bn = self.ur2_c_bn(ur2_c)
        ur2_c_bn_relu = self.ur2_c_bn_relu(ur2_c_bn)
        ur2_c_out = self.ur2_c_out(ur2_c_bn_relu)

        u2_1 = Add()([ur2_c_out, u2_0])
#         u2_out = Concatenate()([u2_1, d2_out])
        u2_out = Dropout(2e-2)(u2_1)


        u3_bn = self.u3_bn(u2_out)
        u3_relu = self.u3_relu(u3_bn)
        u3_ct = self.u3_ct(u3_relu)
        u3_ct_bn = self.u3_ct_bn(u3_ct)
        u3_ct_relu = self.u3_ct_relu(u3_ct_bn)
        u3_01 = self.u3_01(u3_ct_relu)
        u3_02 = self.u3_02(u2_out)
        u3_0 = Add()([u3_01, u3_02])

        ur3_bn = self.ur3_bn(u3_0)
        ur3_relu = self.ur3_relu(ur3_bn)
        ur3_c = self.ur3_c(ur3_relu)
        ur3_c_bn = self.ur3_c_bn(ur3_c)
        ur3_c_bn_relu = self.ur3_c_bn_relu(ur3_c_bn)
        ur3_c_out = self.ur3_c_out(ur3_c_bn_relu)

        u3_1 = Add()([ur3_c_out,u3_0])
#         u3_out = Concatenate()([u3_1,d1_out])
        u3_out = Dropout(2e-2)(u3_1)


        u4_bn = self.u4_bn(u3_out)
        u4_relu = self.u4_relu(u4_bn)
        u4_ct = self.u4_ct(u4_relu)
        u4_ct_bn = self.u4_ct_bn(u4_ct)
        u4_ct_relu = self.u4_ct_relu(u4_ct_bn)
        u4_01 = self.u4_01(u4_ct_relu)
        u4_02 = self.u4_02(u3_out)
        u4_0 = Add()([u4_01, u4_02])

        ur4_bn = self.ur4_bn(u4_0)
        ur4_relu = self.ur4_relu(ur4_bn)
        ur4_c = self.ur4_c(ur4_relu)
        ur4_c_bn = self.ur4_c_bn(ur4_c)
        ur4_c_bn_relu = self.ur4_c_bn_relu(ur4_c_bn)
        ur4_c_out = self.ur4_c_out(ur4_c_bn_relu)

        u4_1 = Add()([ur4_c_out,u4_0])
#         u4_out = Concatenate()([u4_1,basic_3d_res_in])
        u4_out = Dropout(2e-2)(u4_1)


        r1_bn = self.r1_bn(u4_out)
        r1_relu = self.r1_relu(r1_bn)
        r1_c = self.r1_c(r1_relu)
        r1_c_bn = self.r1_c_bn(r1_c)
        r1_c_bn_relu = self.r1_c_bn_relu(r1_c_bn)
        r1_c_out_1 = self.r1_c_out_1(r1_c_bn_relu)
        r1_c_out_2 = self.r1_c_out_2(u4_out)
        r1_out_1 = Add()([r1_c_out_1,r1_c_out_2])

        r1_bn_2 = self.r1_bn_2(r1_out_1)
        r1_relu_2 = self.r1_relu_2(r1_bn_2)
        r1_c_2 = self.r1_c_2(r1_relu_2)
        r1_c_bn_2 = self.r1_c_bn_2(r1_c_2)
        r1_c_bn_relu_2 = self.r1_c_bn_relu_2(r1_c_bn_2)
        r1_c_out_1_2 = self.r1_c_out_1_2(r1_c_bn_relu_2)
        r1_c_out_2_2 = self.r1_c_out_2_2(r1_out_1)
        r1_out_2 = Add()([r1_c_out_1_2,r1_c_out_2_2])
        r1_out = Dropout(2e-2)(r1_out_2)


        r2_bn = self.r2_bn(r1_out)
        r2_relu = self.r2_relu(r2_bn)
        r2_c = self.r2_c(r2_relu)
        r2_c_bn = self.r2_c_bn(r2_c)
        r2_c_bn_relu = self.r2_c_bn_relu(r2_c_bn)
        r2_c_out_1 = self.r2_c_out_1(r2_c_bn_relu)
        r2_c_out_2 = self.r2_c_out_2(r1_out)
        r2_out_1 = Add()([r2_c_out_1,r2_c_out_2])

        r2_bn_2 = self.r2_bn_2(r2_out_1)
        r2_relu_2 = self.r2_relu_2(r2_bn_2)
        r2_c_2 = self.r2_c_2(r2_relu_2)
        r2_c_bn_2 = self.r2_c_bn_2(r2_c_2)
        r2_c_bn_relu_2 = self.r2_c_bn_relu_2(r2_c_bn_2)
        r2_c_out_1_2 = self.r2_c_out_1_2(r2_c_bn_relu_2)
        r2_c_out_2_2 = self.r2_c_out_2_2(r2_out_1)
        r2_out_2 = Add()([r2_c_out_1_2,r2_c_out_2_2])
        r2_out = Dropout(2e-2)(r2_out_2)
        
        return r2_out


# In[53]:


class BiasLayer(Layer):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.reg = 1e-4

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True,
                                    regularizer=regularizers.l2(self.reg))
        
    def call(self, x):
        return x + self.bias


def shape_list(x):
    static = x.get_shape().as_list()
    shape = K.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)

    return ret


def split_heads_2d(inputs, Nh):
    B, H, W, L, d = shape_list(inputs)
    ret_shape = [B, H, W, L, Nh, d//Nh]
    split = K.reshape(inputs, ret_shape)

    return K.permute_dimensions(split, (0,4,1,2,3,5))
    

def combine_heads_2d(inputs):
    transposed = K.permute_dimensions(inputs, (0,2,3,4,1,5))
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]

    return K.reshape(transposed, ret_shape)


class MultiHeadAttention_SingleAxis(Layer):
    def __init__(self, d_model, filters_in, num_heads, attention_axis):
        super(MultiHeadAttention_SingleAxis, self).__init__()
        
        self.d_model = d_model
        self.Nh = num_heads
        self.attention_axis = attention_axis
        self.dropout_rate = 0.1
        self.filters_in = filters_in
        
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

        q = split_heads_2d(self.Wq(x), self.Nh) # B x Nh x H x W x L x dqh
        q *= dqh ** -0.5
        k = split_heads_2d(self.Wk(x), self.Nh)
        v = split_heads_2d(self.Wv(x), self.Nh)
        
        if self.attention_axis == 1:
            q = K.permute_dimensions(q, (0,1,3,4,2,5)) # B x Nh x W x L x H x dqh
            k = K.permute_dimensions(k, (0,1,3,4,2,5))
            v = K.permute_dimensions(v, (0,1,3,4,2,5))
            
        elif self.attention_axis == 2:
            q = K.permute_dimensions(q, (0,1,2,4,3,5)) # B x Nh x H x L x W x dqh
            k = K.permute_dimensions(k, (0,1,2,4,3,5))
            v = K.permute_dimensions(v, (0,1,2,4,3,5))
            
        attn_compat = tf.matmul(q, k, transpose_b = True)
        attn_prob = K.softmax(attn_compat)
        attn_out = tf.matmul(attn_prob, v)
        attn_out = combine_heads_2d(attn_out)
        attn_out = Dropout(self.dropout_rate)(self.Wo(attn_out))
        
        if self.attention_axis == 1:
            attn_out = K.permute_dimensions(attn_out, (0,3,1,2,4))
            
        elif self.attention_axis == 2:
            attn_out = K.permute_dimensions(attn_out, (0,1,3,2,4))
            
        return attn_out


class ConvGRU2D(Layer):
    def __init__(self, filter_size, reg=1e-4):
        super(ConvGRU2D, self).__init__()
        
        self.filter_size = filter_size
        self.kernel_size = 3
        self.strides = (1,1,1)
        self.dilation_rate = (1,1,1)
        
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        self.relu = Activation('relu')
        
        self.Wr = Conv2D(filter_size, (3,3), strides=(1,1), padding='same')
        self.Ur = Conv2D(filter_size, (3,3), strides=(1,1), padding='same')
        self.br = BiasLayer()
        
        self.Wz = Conv2D(filter_size, (3,3), strides=(1,1), padding='same')
        self.Uz = Conv2D(filter_size, (3,3), strides=(1,1), padding='same')
        self.bz = BiasLayer()
        
        self.bh = BiasLayer()
        
        self.W = Conv2D(filter_size, (3,3), strides=(1,1), padding='same')
        self.U = Conv2D(filter_size, (3,3), strides=(1,1), padding='same')
    
    
    def call(self, x, h):
        r = self.sigmoid(self.br(Add()([self.Wr(x), self.Ur(h)])))
        z = self.sigmoid(self.bz(Add()([self.Wz(x), self.Uz(h)])))
        r_x_h = self.U(Multiply()([r, h]))
        th = self.relu(self.bh(Add()([self.W(x), r_x_h])))
        
        ones_tensor = tf.constant(value=1.0, shape=z.shape, dtype=z.dtype)
        cz = Subtract()([ones_tensor, z])
        
        z_x_h = Multiply()([z, h])
        cz_x_th = Multiply()([cz, th])
        
        # h = Add()([self.rbn_z_x_h(z_x_h), self.rbn_cz_x_th(cz_x_th)])
        h = Add()([z_x_h, cz_x_th])
        
        return h


# In[54]:


class RNN_3PIE(Layer):
    def __init__(self, enc_filter, dec_filter, gru_filter, num_rows, num_cols, num_layers):
        super(RNN_3PIE, self).__init__()
        
        self.dropout_rate = 0.1
        
        self.enc_filter = enc_filter
        self.dec_filter = dec_filter
        self.gru_filter = gru_filter
        
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_layers = num_layers
        self.N_h = self.num_rows // 2**4
        
        self.enc = Encoder(enc_filter)
        self.gru_f = ConvGRU2D(self.gru_filter)
        self.gru_b = ConvGRU2D(self.gru_filter)

        self.num_heads = 8
        self.attn = MultiHeadAttention_SingleAxis(self.gru_filter, 2*self.gru_filter, self.num_heads, 3)
        self.ln = LayerNormalization()
        
        self.dec = Decoder(dec_filter, num_layers)

        
    def call(self, x):
        x = Dropout(self.dropout_rate)(self.enc(x))
#         x = K.permute_dimensions(x, (0,3,1,2,4))
        x_ = x[:,:,:,::-1,:]
        
        hf = []
        hb = []
        
        h_f = tf.zeros(shape = (x.shape[0], self.N_h, self.N_h, self.gru_filter))
        h_b = tf.zeros(shape = (x.shape[0], self.N_h, self.N_h, self.gru_filter))
        
        for k in range(self.num_layers):
            x_f = x[:,:,:,k,:]
            h_f = self.gru_f(x_f, h_f)
            hf.append(h_f)
            
            x_b = x_[:,:,:,k,:]
            h_b = self.gru_b(x_b, h_b)
            hb.append(h_b)
        
        hf = K.permute_dimensions(tf.stack(hf), (1,2,3,0,4))
        hb = K.permute_dimensions(tf.stack(hb), (1,2,3,0,4))
        hb = hb[:,:,:,::-1,:]
        h = Concatenate()([hf, hb])
        
        h_att = Dropout(self.dropout_rate)(self.attn(h))
        h_out = self.ln(Add()([h, h_att]))
        h_out = K.sum(h_out, 3)

        x = self.dec(h_out)
        
        return x
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'enc_filter': self.enc_filter,
            'dec_filter': self.dec_filter,
            'gru_filter': self.gru_filter,
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'num_layers': self.num_layers})
        
        return config


# In[55]:
def model_loss(self):
    self.parameters = tf.Variable([0.016 * 0.6, 0.03 * 0.7, 0.04 * 4, 0.54 * 4, 0.027 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.063 * 2, 
                                   0.07 * 2, 0.063 * 2, 0.08 * 2, 0.063 * 2, 0.08 * 2, 0.068 * 2, 0.08 * 2, 0.8 * 2, 0.85 * 2, 0.8 * 2, 0.85 * 2],
                                   trainable = True)
    params = self.parameters
    
    def adaptive_layer_npcc(y_pred, y_true):
        nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
        nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
        nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz

        den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
        den_true = K.std(y_true, axis=(1,2))
        den = K.clip(den_pred * den_true, K.epsilon(), None)

        npcc_loss = (-1) * K.sum(nom / den * params, axis=-1)
    
        return npcc_loss
    
    return adaptive_layer_npcc


def inv_weighted_layer_npcc(y_pred, y_true):
    weights_Nz = tf.constant([0.016 * 0.6, 0.03 * 0.7, 0.04 * 4, 0.54 * 4, 0.027 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.06 * 2, 0.07 * 2, 0.063 * 2,
                              0.07 * 2, 0.063 * 2, 0.08 * 2, 0.063 * 2, 0.08 * 2, 0.068 * 2, 0.08 * 2, 0.8 * 2, 0.85 * 2, 0.8 * 2, 0.85 * 2])
    weights_Nz /= K.sum(weights_Nz) # Nz
    
    nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
    nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
    nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz
    
    den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
    den_true = K.std(y_true, axis=(1,2))
    den = K.clip(den_pred * den_true, K.epsilon(), None)
    
    npcc_loss = (-1) * K.sum(nom / den * weights_Nz, axis=-1)
    
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

def g_loss_npcc(generated_image, true_image):
    fsp=generated_image-K.mean(generated_image,axis=(1,2,3),keepdims=True)
    fst=true_image-K.mean(true_image,axis=(1,2,3),keepdims=True)
    
    devP=K.std(generated_image,axis=(1,2,3))
    devT=K.std(true_image,axis=(1,2,3))
    
    npcc_loss=(-1)*K.mean(fsp*fst,axis=(1,2,3))/K.clip(devP*devT,K.epsilon(),None)    ## (BL,1)
    return npcc_loss


# In[56]:


# # Model definition
# # 1. Encoder from PhENN - we will use measurements as inputs
# enc_filter = [32, 64, 96, 128, 256, 256]
# dec_filter = [48, 64, 128, 256, 512]
# gru_filter = 512
# num_rows = 128
# num_cols = 128
# num_layers = 22

# # strategy = tf.distribute.MirroredStrategy()
# # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# # with strategy.scope():
# x = Input(shape = (128, 128, 22, 1), batch_size = 10)
# out = RNN_3PIE(enc_filter, dec_filter, gru_filter, num_rows, num_cols, num_layers)(x)
# model = Model(x, out)
# model.summary()

# optadam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
# optdelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.0)
# model.compile(optimizer=optdelta,loss=model_loss, metrics=[layer_npcc])


# In[ ]:


# In[37]:


# Model definition
# 1. Encoder from PhENN - we will use measurements as inputs
enc_filter = [32, 64, 96, 128, 256, 256]
dec_filter = [48, 64, 128, 256, 512]
gru_filter = 512
num_rows = 128
num_cols = 128
num_layers = 22

#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#with strategy.scope():
def adaptive_layer_npcc(y_true, y_pred, weights):
    nom_pred = y_pred - K.mean(y_pred, axis=(1,2), keepdims=True) # B x Ny x Nx x Nz - B x 1 x 1 x Nz = B x Ny x Nx x Nz
    nom_true = y_true - K.mean(y_true, axis=(1,2), keepdims=True)
    nom = K.mean(nom_pred * nom_true, axis=(1,2)) # B x Nz

    den_pred = K.std(y_pred, axis=(1,2)) # B x Nz
    den_true = K.std(y_true, axis=(1,2))
    den = K.clip(den_pred * den_true, K.epsilon(), None)

    npcc_loss = (-1) * K.sum(nom / den * weights, axis=-1)

    return npcc_loss

x = Input(shape = (128, 128, 22, 1), batch_size = 10)
true = Input(shape = (128, 128, 22), batch_size = 10)
out = RNN_3PIE(enc_filter, dec_filter, gru_filter, num_rows, num_cols, num_layers)(x)
params = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable = True)

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

wp = wrapped_partial(adaptive_layer_npcc, weights = params)
model = Model(x, out)
model.summary()

optadam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
optdelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.0)
model.compile(optimizer=optdelta, loss=wp, metrics = [layer_npcc])
    
    
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

#training now!
class LearningRateBasedStopping(tf.keras.callbacks.Callback):
    def __init__(self, limit_lr):
        super(LearningRateBasedStopping, self).__init__()
        self.limit_lr = limit_lr
	
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, lr))
                                    
        if lr < self.limit_lr:
            self.model.stop_training = True

            
batch_size = 10
num_epochs = 200

weight_folder = 'weights/attn_rnn_3pie_v1_adaptive_/'
if not os.path.exists(weight_folder):
    os.makedirs(weight_folder)

csv_logger = CSVLogger('log/attn_rnn_3pie_v1_adaptive_.log')
checkpoint = ModelCheckpoint(filepath = weight_folder + '{epoch:02d}.hdf5',
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
lr_stopping = LearningRateBasedStopping(1e-8)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001,
                             cooldown=0, min_lr=1e-8)
callbacks_list = [checkpoint, csv_logger, lr_stopping, reducelr]

model.load_weights('weights/attn_rnn_3pie_v1_adaptive_/25.hdf5')

#model.fit(training_input, training_output, batch_size = batch_size, epochs = num_epochs,
#          validation_data = (validation_input, validation_output), verbose = 1, callbacks = callbacks_list)

experiment = np.load('experiment_approx_layer.npy')
experiment = experiment[:, :, :, 0] + 1j * experiment[:, :, :, 1]
experiment = np.expand_dims(experiment[192:320, 192:320, :], axis=0)
print(np.shape(experiment))


rec = model.predict(experiment, batch_size = 1, verbose =1)

np.save('rec_attn_3pie_layer', rec)
    
