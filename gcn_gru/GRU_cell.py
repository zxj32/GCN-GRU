from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.platform import tf_logging as logging
from gcn_gru.layers import GraphConvolution, Graph_matual, GraphConvolution_saprse


class GCNGRUCell_sparse(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def _compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, dropout, num_nodes, index, value, bias, input_size=None, reuse=None):
        """
        :param num_units:
        :param adj_mx:
        :param num_nodes:
        :param input_size:
        :param reuse:
        """
        super(GCNGRUCell_sparse, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_nodes = num_nodes
        self._num_units = num_units
        self.adj = adj_mx
        self.dropout = dropout
        self.logging = logging
        self.bias = bias
        # self.index = index
        # self.value = value
        self.input_rnn = tf.SparseTensor(index, value, [num_nodes, num_nodes])


    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            # inputs = tf.ones([self._num_nodes, self._num_nodes])
            # x_shape = tf.concat([inputs, state], axis=1)
            # input_shape = x_shape.get_shape().as_list()
            # input_rnn = tf.SparseTensor(self.index, self.value, [self._num_nodes, self._num_nodes])
            state = tf.reshape(state, [self._num_nodes, -1])
            state_shape = state.get_shape().as_list()
            input_dim = state_shape[1] + self._num_nodes

            zero = tf.constant(0, dtype=tf.float32)
            where = tf.not_equal(state, zero)
            indices = tf.where(where)
            values = tf.gather_nd(state, indices)
            sparse_state = tf.SparseTensor(indices, values, state.shape)
            x1 = tf.sparse_concat(sp_inputs=[self.input_rnn, sparse_state], axis=1)
            non_zeros_feat = tf.size(x1.values)
            with tf.variable_scope("gates"):  # Reset gate
                # We start with bias of 1.0 to not reset and not update.
                r = GraphConvolution_saprse(input_dim=input_dim,
                                     output_dim=self._num_units,
                                     adj=self.adj,
                                     act=tf.nn.sigmoid,
                                     dropout=self.dropout,
                                     non_zero = non_zeros_feat,
                                     bias=self.bias,
                                     logging=self.logging)(x1)

            with tf.variable_scope("gates"):  # Update gate
                u = GraphConvolution_saprse(input_dim=input_dim,
                                     output_dim=self._num_units,
                                     adj=self.adj,
                                     act=tf.nn.sigmoid,
                                     dropout=self.dropout,
                                     non_zero=non_zeros_feat,
                                     bias=self.bias,
                                     logging=self.logging)(x1)

            with tf.variable_scope("candidate"):
                state_r = r * state
                where_r = tf.not_equal(state_r, zero)
                indices_r = tf.where(where_r)
                values_r = tf.gather_nd(state_r, indices_r)
                sparse_state_r = tf.SparseTensor(indices_r, values_r, state_r.shape)
                x2 = tf.sparse_concat(sp_inputs=[self.input_rnn, sparse_state_r], axis=1)

                c = GraphConvolution_saprse(input_dim=input_dim,
                                     output_dim=self._num_units,
                                     adj=self.adj,
                                     act=tf.nn.sigmoid,
                                     dropout=self.dropout,
                                     non_zero=non_zeros_feat,
                                     bias=self.bias,
                                     logging=self.logging)(x2)

            h = u * state + (1 - u) * c
            h = tf.reshape(h, [1, -1])
            output = new_state = h
        return output, new_state


class GCNGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def _compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, dropout, num_nodes, bias, input_size=None, reuse=None):
        """
        :param num_units:
        :param adj_mx:
        :param num_nodes:
        :param input_size:
        :param reuse:
        """
        super(GCNGRUCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_nodes = num_nodes
        self._num_units = num_units
        self.adj = adj_mx
        self.dropout = dropout
        self.logging = logging
        self.bias = bias

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            inputs = tf.reshape(inputs, [self._num_nodes, -1])
            state = tf.reshape(state, [self._num_nodes, -1])
            x1 = tf.concat([inputs, state], axis=1)
            input_shape = x1.get_shape().as_list()
            with tf.variable_scope("gates"):  # Reset gate
                # We start with bias of 1.0 to not reset and not update.
                r = GraphConvolution(input_dim=input_shape[1],
                                     output_dim=self._num_units,
                                     adj=self.adj,
                                     act=tf.nn.sigmoid,
                                     dropout=self.dropout,
                                     bias=self.bias,
                                     logging=self.logging)(x1)

            with tf.variable_scope("gates"):  # Update gate
                u = GraphConvolution(input_dim=input_shape[1],
                                     output_dim=self._num_units,
                                     adj=self.adj,
                                     act=tf.nn.sigmoid,
                                     dropout=self.dropout,
                                     bias=self.bias,
                                     logging=self.logging)(x1)

            with tf.variable_scope("candidate"):
                x2 = tf.concat([inputs, r * state], axis=1)
                c = GraphConvolution(input_dim=input_shape[1],
                                     output_dim=self._num_units,
                                     adj=self.adj,
                                     act=tf.nn.sigmoid,
                                     dropout=self.dropout,
                                     bias=self.bias,
                                     logging=self.logging)(x2)

            h = u * state + (1 - u) * c
            h = tf.reshape(h, [1, -1])
            output = new_state = h
        return output, new_state