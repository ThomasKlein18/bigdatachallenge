import collections
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers 
from tensorflow.python.ops import init_ops
from autoconceptor import Autoconceptor

class AutoconLayer(layers.RNN):

    def __init__(self, output_dim, alpha, lam, batchsize, activation=tf.nn.tanh, layer_norm=False, reuse=None, **kwargs):
        self.output_dim = output_dim
        self._cell = Autoconceptor(output_dim, alpha, lam, batchsize, 
            activation=tf.nn.tanh, reuse=reuse, layer_norm=layer_norm, dtype=tf.float32, 
            initializer=None)
        super(AutoconLayer, self).__init__(cell=self._cell, **kwargs)

    def build(self, input_shape):
        print("input shape:", input_shape)
        # Make sure to call the `build` method at the end
        self._cell.build(input_shape)
        #super(AutoconLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(AutoconLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)