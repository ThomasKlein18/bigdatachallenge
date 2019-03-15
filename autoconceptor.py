"""
The Autoconceptor, adapted from Jaeger 2017, and the DynStateTuple that
is used to store the conceptor matrix.
"""

import numpy as np
import collections
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.layers import base as base_layer

# following the desing of LSTM state tuples
_DynStateTuple = collections.namedtuple("DynStateTyple", ["C", "h"])

class DynStateTuple(_DynStateTuple):
    """Tuple used by RNN Models with conceptor matrices.

    Stores two elements: `(C, h)` in that order
        where C is the conceptor matrix
        and   h is the state of the RNN

    adapted from LSTMStateTuple in tensorflow/python/obs/rnn_cell_impl.py
    """

    __slots__ = ()

    @property
    def dtype(self):
        (C, h) = self
        if C.dtype != h.dtype:
            raise TypeError("Matrix and internal state should agree on type: %s vs %s" %
                            (str(C.dtype), str(h.dtype)))
        return C.dtype


class Autoconceptor(tf.nn.rnn_cell.BasicRNNCell):
    """
    Autoconceptor, adapted from Jaeger 2017
    """

    def __init__(self, num_units, alpha, lam, batchsize, 
            activation=tf.nn.tanh, reuse=None, layer_norm=False, dtype=tf.float32, 
            initializer=None):
        """
        Args:
        num_units   = hidden state size of RNN cell
        alpha       = alpha for autoconceptor, used to calculate aperture as alpha**-2
        lam         = lambda for autoconceptor, scales conceptor-matrix
        batchsize   = number of training examples per batch (we need this to allocate memory properly)
        activation  = which nonlinearity to use (tanh works best, relu only with layer norm)
        reuse       = whether to reuse variables, just leave this as None
        layer_norm  = whether to apply layer normalization, not necessary if using tanh
        initializer = which initializer to use for the weight matrix, good idea is to use init_ops.constant_initializer(0.05 * np.identity(num_units))
        """
        super(Autoconceptor, self).__init__(num_units=num_units, activation=activation, reuse=reuse)
        self.num_units = num_units
        self.c_lambda = tf.constant(lam, name="lambda")
        self.batchsize = batchsize
        self.conceptor_built = False
        self.layer_norm = layer_norm
        self._activation = activation
        self.aperture_fact = tf.constant(alpha**(-2), name="aperture")
        self._state_size = self.zero_state(batchsize, dtype)
        self.initializer = initializer or init_ops.constant_initializer(0.05 * np.identity(num_units))

        #no idea what this does, to be honest
        self.input_spec = base_layer.InputSpec(ndim=2)

    # these two properties are necessary to pass assert_like_rnn_cell test in static_rnn and dynamic_rnn
    @property
    def state_size(self):
        
        return  self._state_size

    @property
    def output_size(self):
        return self.num_units

    def zero_state(self, batch_size, dtype):
        """
        Returns the zero state for the autoconceptor cell.

        batch_size = the number of elements per batch
        dtype      = the dtype to be used, stick with tf.float32

        The zero state is a DynStateTuple consisting of a C-matrix filled with zeros,
        shape [batchsize, num_units, num_units] and a zero-filled hidden state of
        shape [batchsize, num_units]
        """
        return DynStateTuple(C=tf.zeros([batch_size, self.num_units, self.num_units], dtype=dtype),
                             h=tf.zeros([batch_size, self.num_units], dtype=dtype))


    def build(self, inputs_shape):
        """
        Builds the cell by defining variables. 
        Overrides method from super-class.
        """
        print("inputs shape at autoconceptor: ", inputs_shape) # None, 80, 19
        if inputs_shape[2] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
        input_dim = inputs_shape[2]

        self.W_in = self.add_variable(
            "W_in",
            shape=[input_dim, self.num_units],
            initializer=init_ops.random_normal_initializer(),
            dtype=self.dtype)

        self.b_in = self.add_variable(
            "b_in",
            shape=[self.num_units],
            initializer= init_ops.zeros_initializer(),
            dtype=self.dtype)

        self.W = self.add_variable(
            "W",
            shape=[self.num_units, self.num_units],
            initializer=self.initializer,
            dtype=self.dtype)

        
        #tf.get_variable("gamma", shape=shape, initializer=gamma_init)
        #    tf.get_variable("beta", shape=shape, initializer=beta_init)

        self.built = True

    
    # def _norm(self, inp, scope="layer_norm"):
    #     """ 
    #     Performs layer normalization on the hidden state.

    #     inp = the input to be normalized
    #     scope = name for the variable scope, just leave as default
        
    #     Returns inp normalized by learned parameters gamma and beta
    #     """
    #     #shape = inp.get_shape()[-1:]
    #     #gamma_init = init_ops.constant_initializer(1)
    #     #beta_init = init_ops.constant_initializer(1)
    #     #with tf.variable_scope(scope):
    #     #    tf.get_variable("gamma", shape=shape, initializer=gamma_init)
    #     #    tf.get_variable("beta", shape=shape, initializer=beta_init)
    #     normalized = layers.layer_norm(inp)
    #     return normalized


    def call(self, inputs, h):
        """
        Performs one step of this Autoconceptor Cell.

        inputs = the input batch, shape [batchsize, input_dim]
        h      = the DynStateTuple containing the preceding state

        Returns output, state
            where output = output at this time step
                  state  = new hidden state and C-matrix as DynStateTuple
        """

        print("inputs in call, should be 32x19:",inputs)
        
        C, state = h

        print("C in call, should be 32x50x50:", C)
        print("State in call, should be 32x50, I guess:", h)
        # so far, standard RNN logic
        state = self._activation(
            (tf.matmul(inputs, self.W_in) + self.b_in) + (tf.matmul(state, self.W))
        )

        # if layer norm is activated, normalize layer output as explained in Ba et al. 2016
        if(self.layer_norm):
            state = layers.layer_norm(state)#self._norm(state)
        
        state = tf.reshape(state, [-1, 1, self.num_units])

        

        # updating C following update rule presented by Jaeger
        C = C + self.c_lambda * ( tf.matmul(tf.transpose((state - tf.matmul(state, C)), [0,2,1]), state) - tf.scalar_mul(self.aperture_fact,C) )
        
        # multiplying state with C
        state = tf.matmul(state, C)

        # Reshapes necessary for std. matrix multiplication, where one matrix
        # for all elements in batch vs. fast-weights matrix -> different for every
        # element!
        state = tf.reshape(state, [-1, self.num_units])

        return state, DynStateTuple(C, state)
