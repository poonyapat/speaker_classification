#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import numpy  as np
import tensorflow as tf
import keras
from keras.layers import *
from scipy.io import wavfile
import os
from tqdm import tqdm
import threading
import glob
import gc
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import audio_features_utils
from IPython.display import Audio
from PIL import Image
import pandas as pd
import seaborn as sns

# In[3]:


def to255(x):
    temp = x - x.min()
    return (temp*255/temp.max()).astype(np.uint8)
def iswish(x):
    return x-swish(x)
def swish(x): 
    return keras.backend.sigmoid(x)*x
def sigmoid2(x):
    return keras.activations.sigmoid(x)+1e-10
def euclidean_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred),axis=-1))*100
def euclidean_layer(x1, x2):
    distance = tf.sqrt(tf.reduce_sum(tf.square(x1 - x2),axis=-1))
    return tf.expand_dims(distance, axis=-1)
def euclidean_numpy(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2),axis=-1))
def ExpandDims(axis=-1):
    return Lambda(lambda x: tf.expand_dims(x, axis=axis))
def Squeeze(axis=-1):
    return Lambda(lambda x: tf.squeeze(x, axis=axis))
def Split(n=2, axis=-1):
    return Lambda(lambda x: tf.split(x, n, axis=axis))
# def Mean():
#     return Lambda(lambda x: )
def sdense(unit, x):
    inner = Dense(unit)(x)
    inner = BatchNormalization()(inner)
    inner = Activation(swish)(inner)
    return inner
def combine(inputs):
    for i in range(len(inputs)):
        inputs[i] = ExpandDims(-1)(inputs[i])
    inner = Concatenate(axis=-1)(inputs)
    inner = ExpandDims(-1)(inner)
    inner = LocallyConnected2D(1, (1,len(inputs)))(inner)
    inner = Squeeze((-1, -2))(inner)
    return inner
def gate(inp, activation='softmax'):
    f = inp.shape.as_list()[-1]
    return Dense(f, activation=activation)(inp)

def gated(inp, activation='softmax'):
    return Multiply()([gate(inp, activation), inp])

def abswish(x):
    return swish(x)+0.27847
    
class Logger(keras.callbacks.Callback):
    
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.c = 0
        self.logs = dict()
    
    def monitor_text(self):
        out = ''
        for key in self.logs:
            out += ', {}: {:7.3f}'.format(key, self.logs[key]/self.c)
            self.logs[key] = 0
        self.c = 0
        return out
    
    def monitor_text2(self, logs):
        out = ''
        for key in logs:
            if key == 'batch' or key == 'size':
                continue
            out += ', {}: {:7.3f}'.format(key, logs[key])
            self.logs[key] = 0
        self.c = 0
        return out
        
    def on_epoch_end(self, epoch, logs=None):
        print('Epoch {} {}'.format(epoch, self.monitor_text2(logs), end='\r'))
        
    def update_logs(self, logs):
        for key in logs:
            if key == 'batch' or key == 'size':
                continue
            if key in self.logs:
                self.logs[key] += logs[key]
            else:
                self.logs[key] = logs[key]
        self.c+=1

        
    def on_batch_end(self, batch, logs=None):
        self.update_logs(logs)
        if not batch%self.checkpoint:
            print('For batch {} on {}{}'.format(batch, batch*logs['size'], self.monitor_text(), end='\r'))


# In[ ]:


# Utils
def framing1D(seq, kernel_size, strides):
    si = np.arange(0,len(seq),strides)
    ei = si+kernel_size
    indexes = np.linspace(si,ei,kernel_size, dtype=np.uint64).T
    pad_seq = np.pad(seq, (0, kernel_size+1-len(seq)%strides), 'constant', constant_values=0)
    return pad_seq[indexes]

def count_frames(seq, strides):
    return np.ceil(len(seq)/strides).astype(np.uint16)

