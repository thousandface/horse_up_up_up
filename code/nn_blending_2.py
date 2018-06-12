# coding: utf-8

import os
import re
import sys  
import pandas as pd
import numpy as np

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import tensorflow as tf

import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils.training_utils import multi_gpu_model

from sklearn.cross_validation import train_test_split

from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def get_text_capsule(sent_length, embeddings_weight):
    print ("get_text_capsule")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    embed = SpatialDropout1D(0.2)(embedding(content))
    
    x = Bidirectional(CuDNNGRU(128, return_sequences = True))(embed)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,share_weights=True)(x)
    capsule = Flatten()(capsule)
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)( capsule))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def get_text_lstm3(sent_length, embeddings_weight):
    print ("get_text_lstm3")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed))
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model    

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)
    
def get_text_rcnn4(sent_length, embeddings_weight):
    print ("get_text_rcnn4")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))

    rnn_1 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embed)
    conv_2 = Conv1D(128, 2, kernel_initializer="normal", padding="valid", activation="relu", strides=1)(rnn_1)
    
    maxpool = GlobalMaxPooling1D()(conv_2)
    attn = AttentionWeightedAverage()(conv_2)
    average = GlobalAveragePooling1D()(conv_2)

    x = concatenate([maxpool, attn, average])
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


def get_text_rcnn5(sent_length, embeddings_weight):
    print ("get_text_rcnn5")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))

    rnn_1 = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    rnn_2 = Bidirectional(CuDNNGRU(200, return_sequences=True))(rnn_1)
    x = concatenate([rnn_1, rnn_2], axis=2)

    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    average = GlobalAveragePooling1D()(x)

    x= concatenate([last, maxpool, average, attn])
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model
    
def get_text_gru5(sent_length, embeddings_weight):
    print ("get_text_gru5")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))
                  
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    x = Dropout(0.35)(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    last = Lambda(lambda t: t[:, -1])(x)
    maxpool = GlobalMaxPooling1D()(x)
    average = GlobalAveragePooling1D()(x)
    x = concatenate([last, maxpool, average])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def get_text_cnn1(sent_length, embeddings_weight):
    print ("get_text_cnn1")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
     input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    embed = embedding(content)

    embed = SpatialDropout1D(0.2)(embed)
    
    conv2 = Activation('relu')(BatchNormalization()(Conv1D(128, 2, padding='same')(embed)))
    conv2 = Activation('relu')(BatchNormalization()(Conv1D(64, 2, padding='same')(conv2)))
    conv2 = MaxPool1D(pool_size=50)(conv2)

    conv3 = Activation('relu')(BatchNormalization()(Conv1D(128, 3, padding='same')(embed)))
    conv3 = Activation('relu')(BatchNormalization()(Conv1D(64, 3, padding='same')(conv3)))
    conv3 = MaxPool1D(pool_size=50)(conv3)
    
    conv4 = Activation('relu')(BatchNormalization()(Conv1D(128, 4, padding='same')(embed)))
    conv4 = Activation('relu')(BatchNormalization()(Conv1D(64, 4, padding='same')(conv4)))
    conv4 = MaxPool1D(pool_size=50)(conv4)

    conv5 = Activation('relu')(BatchNormalization()(Conv1D(128, 5, padding='same')(embed)))
    conv5 = Activation('relu')(BatchNormalization()(Conv1D(64, 5, padding='same')(conv5)))
    conv5 = MaxPool1D(pool_size=50)(conv5)
    
    cnn = concatenate([conv2, conv3, conv4, conv5], axis=-1)
    flat = Flatten()(cnn)

    drop = Dropout(0.2)(flat)
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(drop))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# In[35]:

def get_text_cnn2(sent_length, embeddings_weight):
    print ("get_text_cnn2")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    embed = embedding(content)
    filter_sizes = [1,2,3,4]
    num_filters = 128
    embed_size = embeddings_weight.shape[1]

    x = SpatialDropout1D(0.2)(embed)
    x = Reshape((sent_length, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(sent_length - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sent_length - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sent_length - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(sent_length - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(z))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# In[36]:

def get_text_cnn3(sent_length, embeddings_weight):
    print ("get_text_cnn3")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)(content)

    embedding = SpatialDropout1D(0.2)(embedding)
    
    cnn1 = Conv1D(128, 2, padding='same', strides=1, activation='relu')(embedding)
    cnn2 = Conv1D(128, 3, padding='same', strides=1, activation='relu')(embedding)
    cnn3 = Conv1D(128, 4, padding='same', strides=1, activation='relu')(embedding)
    cnn4 = Conv1D(128, 5, padding='same', strides=1, activation='relu')(embedding)
    cnn = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)

    cnn1 = Conv1D(64, 2, padding='same', strides=1, activation='relu')(cnn)
    cnn1 = MaxPooling1D(pool_size=100)(cnn1)
    cnn2 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(cnn)
    cnn2 = MaxPooling1D(pool_size=100)(cnn2)
    cnn3 = Conv1D(64, 4, padding='same', strides=1, activation='relu')(cnn)
    cnn3 = MaxPooling1D(pool_size=100)(cnn3)
    cnn4 = Conv1D(64, 5, padding='same', strides=1, activation='relu')(cnn)
    cnn4 = MaxPooling1D(pool_size=100)(cnn4)
    
    cnn = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)

    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(drop))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# In[37]:

def get_text_gru1(sent_length, embeddings_weight):
    print ("get_text_gru1")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    x = SpatialDropout1D(0.2)(embedding(content))
    
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# In[47]:

def get_text_gru2(sent_length, embeddings_weight):
    print ("get_text_gru2")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
       name="word_embedding",
       input_dim=embeddings_weight.shape[0], 
       weights=[embeddings_weight], 
       output_dim=embeddings_weight.shape[1], 
       trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))
    
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    
    x = Conv1D(100, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model 


# In[48]:

def get_text_gru3(sent_length, embeddings_weight):
    print ("get_text_gru3")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
       name="word_embedding",
       input_dim=embeddings_weight.shape[0], 
       weights=[embeddings_weight], 
       output_dim=embeddings_weight.shape[1], 
       trainable=False)
    
    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    att_pool =  Attention(sent_length)(x) 
    conc = concatenate([avg_pool, max_pool, att_pool])
   
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def get_text_gru4(sent_length, embeddings_weight):
    print ("get_text_gru4")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
       name="word_embedding",
       input_dim=embeddings_weight.shape[0], 
       weights=[embeddings_weight], 
       output_dim=embeddings_weight.shape[1], 
       trainable=False)
    x = SpatialDropout1D(0.2)(embedding(content))
    
    x = Bidirectional(CuDNNLSTM(200, return_sequences = True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences = True))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool,max_pool])
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# In[49]:

def get_text_rcnn1(sent_length, embeddings_weight):
    print ("get_text_rcnn1")
    document = Input(shape = (None, ), dtype = "int32")
    
    embedder = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    doc_embedding =  SpatialDropout1D(0.2)(embedder(document))
    forward = Bidirectional(CuDNNLSTM(200, return_sequences = True))(doc_embedding)
    together = concatenate([forward, doc_embedding], axis = 2) 

    semantic =  Conv1D(100, 2, padding='same', strides = 1, activation='relu')(together)
    pool_rnn = Lambda(lambda x: K.max(x, axis = 1), output_shape = (100, ))(semantic) 
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(pool_rnn))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs= document, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def get_text_rcnn2(sent_length, embeddings_weight):
    print ("get_text_rcnn2")
    content = Input(shape = (None, ), dtype = "int32")
    
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    x =  SpatialDropout1D(0.2)(embedding(content))
    
    x = Convolution1D(filters=256,kernel_size=3,padding='same',strides=1,activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.2)(CuDNNGRU(units=200, return_sequences=True)(x))
    x = Dropout(0.2)(CuDNNGRU(units=100)(x))
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def get_text_rcnn3(sent_length, embeddings_weight):
    print ("get_text_rcnn3")
    content = Input(shape = (None, ), dtype = "int32")
    
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    x =  SpatialDropout1D(0.2)(embedding(content))
    
    cnn = Convolution1D(filters=200, kernel_size=3, padding="same", strides=1, activation="relu")(x)
    cnn_avg_pool = GlobalAveragePooling1D()(cnn)
    cnn_max_pool = GlobalMaxPooling1D()(cnn)
    
    
    rnn = Dropout(0.2)(CuDNNGRU(200, return_sequences=True)(x))
    rnn_avg_pool = GlobalAveragePooling1D()(rnn)
    rnn_max_pool = GlobalMaxPooling1D()(rnn)
    
    con = concatenate([cnn_avg_pool, cnn_max_pool, rnn_avg_pool, rnn_max_pool], axis=-1)
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(con))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


def get_text_lstm1(sent_length, embeddings_weight):
    print ("get_text_lstm1")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed))
    semantic = TimeDistributed(Dense(100, activation = "tanh"))(x) 
    pool_rnn = Lambda(lambda x: K.max(x, axis = 1), output_shape = (100, ))(semantic) 
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(pool_rnn))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# In[44]:

def get_text_lstm2(sent_length, embeddings_weight):
    print ("get_text_lstm2")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)) 
    semantic = TimeDistributed(Dense(100, activation = "tanh"))(x) 
    pool_rnn = Lambda(lambda x: K.max(x, axis = 1), output_shape = (100, ))(semantic)
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(pool_rnn))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


def get_text_lstm_attention(sent_length, embeddings_weight):
    print ("get_text_lstm_attention")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embedded_sequences= SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.25)(CuDNNLSTM(200, return_sequences=True)(embedded_sequences))
    merged = Attention(sent_length)(x)
    merged = Dense(100, activation='relu')(merged)
    merged = Dropout(0.25)(merged)
    
    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(merged))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


def get_text_dpcnn(sent_length, embeddings_weight):
    print ("get_text_dpcnn")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))

    block1 = Conv1D(128, kernel_size=3, padding='same', activation='linear')(embed)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(128, kernel_size=3, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    resize_emb = Conv1D(128, kernel_size=3, padding='same', activation='linear')(embed)
    resize_emb = PReLU()(resize_emb)
        
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=10)(block1_output)

    block2 = Conv1D(128, kernel_size=4, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(128, kernel_size=4, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
        
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=10)(block2_output)

    block3 = Conv1D(128, kernel_size=5, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(128, kernel_size=5, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)

    output = add([block3, block2_output])
    maxpool = GlobalMaxPooling1D()(output)
    average = GlobalAveragePooling1D()(output)
    
    x = concatenate([maxpool, average])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def get_text_gru6(sent_length, embeddings_weight):
    print ("get_text_gru6")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0], 
        weights=[embeddings_weight], 
        output_dim=embeddings_weight.shape[1], 
        trainable=False)
    
    embed = SpatialDropout1D(0.2)(embedding(content))
                  
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    x = Conv1D(60, kernel_size=3, padding='valid', activation='relu', strides=1)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    embed = SpatialDropout1D(0.2)(embedding(content))
    y = Bidirectional(CuDNNGRU(100, return_sequences=True))(embed)
    y = Conv1D(40, kernel_size=3, padding='valid', activation='relu', strides=1)(y)
    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)    
    
    x = concatenate([avg_pool, max_pool, avg_pool2,  max_pool2], -1)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output  = Dense(372, activation="softmax")(x)
    
    model = Model(inputs=content, outputs=output)
    model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model


# coding:utf-8

###分词模块
def performance(f):                                                  #定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数  
    def fn(*args, **kw):                                             #对传进来的函数进行包装的函数  
        t_start = time.time()                                        #记录函数开始时间   
        r = f(*args, **kw)                                           #调用函数  
        t_end = time.time()                                          #记录函数结束时间   
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start))) #打印调用函数的属性信息，并打印调用函数所用的时间  
        return r                                                     #返回包装后的函数      
    return fn     

##########################  read data ####################################


train_path  = '../input/msxf_dialog_train.csv'
test_path =   '../input/msxf_dialog_test_2round.csv'
test_path_ =   '../input/msxf_dialog_test_1round.csv'

df_train = pd.read_csv(train_path, sep='\t')
df_test  = pd.read_csv(test_path, sep='\t')
df_test_  = pd.read_csv(test_path_, sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
lb = LabelEncoder()
train_label = lb.fit_transform(df_train["label"].values)
train_label = to_categorical(train_label)

word_seq_len = 100
    
def w2v_pad(col, maxlen_):
    max_features = 150000
    tokenizer = text.Tokenizer(num_words=max_features, lower=True)
    tokenizer.fit_on_texts(list(df_train[col].values)+list(df_test_[col].values))

    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].values), maxlen=maxlen_)
    test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].values), maxlen=maxlen_)
    
    word_index = tokenizer.word_index
    
    count = 0
    nb_words = len(word_index)

    model = gensim.models.KeyedVectors.load_word2vec_format("../input/msxf_dialog_word_embeddings.vec")
            
    embedding_matrix = np.zeros((nb_words + 1, 200))
    
    for word, i in word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(200) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec
    
    print (embedding_matrix.shape, train_.shape, test_.shape, count * 1.0 / embedding_matrix.shape[0])
    return train_, test_, word_index, embedding_matrix


X_train, X_test, word2idx, word_embedding = w2v_pad('question', word_seq_len)


early_stopping =EarlyStopping(monitor='val_loss', patience=6)
plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.5, patience=3)


X_train_ , X_valid, y_train, y_valid = train_test_split(X_train, train_label, random_state=123, train_size=0.95)

def word_model(model):
    name = str(model.__name__)
    file_path= "../models_2/" + name + "_word_weights.hdf"
    model = model(word_seq_len, word_embedding)
    if not os.path.exists(file_path):
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        model.fit(X_train_, y_train,
                    epochs = 100,
                    batch_size=256,
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping, plateau, checkpoint])
    model.load_weights(file_path)
    
    pred_word, pred_test_word = lb.inverse_transform(np.argmax(model.predict(X_valid), 1)).reshape(-1,1), \
        lb.inverse_transform(np.argmax(model.predict(X_test), 1)).reshape(-1,1)
    del model; gc.collect()
    K.clear_session()
    print (pred_word.shape, pred_test_word.shape)
    print (name + ": valid's accuracy: %s" % accuracy_score(lb.inverse_transform(np.argmax(y_valid, 1)), pred_word))
    return (pred_word, pred_test_word)


nn_train, nn_test = zip(*[word_model(get_text_lstm3),\
                          word_model(get_text_rcnn4),\
                          word_model(get_text_rcnn5),\
                          word_model(get_text_gru5),\
                          word_model(get_text_rcnn1),\
                          word_model(get_text_rcnn2),\
                          word_model(get_text_rcnn3),\
                          word_model(get_text_cnn1), \
                          word_model(get_text_cnn2), \
                          word_model(get_text_cnn3), \
                          word_model(get_text_gru1), \
                          word_model(get_text_gru2), \
                          word_model(get_text_gru3), \
                          word_model(get_text_gru4), \
                          word_model(get_text_lstm1), \
                          word_model(get_text_lstm2),
                          word_model(get_text_lstm_attention)])

nn_train =  np.concatenate(nn_train, 1)
nn_test = np.concatenate(nn_test, 1)

np.savez('../blending/round2_nn_blending_2_.npz', train=nn_train, test=nn_test)

from scipy import stats
df_test['label'] = pd.DataFrame(nn_test).apply(lambda x: stats.mode(x)[0][0], axis=1)
df_test[['conv_index', 'question_id', 'label']].to_csv('../output/blending_vote_2.csv', sep='\t', index=None)