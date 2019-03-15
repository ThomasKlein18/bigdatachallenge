import numpy as np 
import tensorflow as tf 
import tensorflow.keras.layers as layers 
from AutoconLayer import AutoconLayer

def get_lazarus(sequence_length):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(60, input_shape=[sequence_length,19]))
    model.add(layers.Dropout(0.2))
    #model.add(LSTM(50))
    model.add(layers.Dense(32, activation='tanh'))
    #model.add(layers.Dropout(0.8))
    model.add(layers.Dense(22, activation='softmax'))
    return model

def get_nathanael(sequence_length):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(60, input_shape=[sequence_length,19]))
    model.add(layers.Dropout(0.5))
    #model.add(LSTM(50))
    model.add(layers.Dense(32, activation='tanh'))
    #model.add(layers.Dropout(0.8))
    model.add(layers.Dense(22, activation='softmax'))
    return model

def get_ptolemaeus(sequence_length):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(60, input_shape=[sequence_length,19]))
    model.add(layers.Dropout(0.8))
    #model.add(LSTM(50))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dropout(0.8))
    model.add(layers.Dense(22, activation='softmax'))
    return model

def get_grindelwald(sequence_length):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(80, input_shape=[sequence_length, 19]))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(layers.Dense(22, activation='softmax'))
    return model

def get_autoconceptor(sequence_length):
    model = tf.keras.Sequential()
    model.add(AutoconLayer(output_dim=50, alpha=200, lam=0.001, batchsize=32, layer_norm=True, reuse=None)) 
    model.add(layers.Dense(32, activation='tanh'))
    #model.add(layers.Dropout(0.8))
    model.add(layers.Dense(22, activation='softmax'))

    return model