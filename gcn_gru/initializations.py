import tensorflow as tf
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    # initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
    #                             maxval=init_range, dtype=tf.float32)
    W = tf.get_variable(name=name, shape=[input_dim, output_dim], initializer=tf.random_uniform_initializer(minval=-init_range,
                                maxval=init_range), dtype=tf.float32)
    return W


def weight_variable_bisa(output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    W = tf.get_variable(name=name, shape=output_dim, initializer=tf.zeros_initializer, dtype=tf.float32)
    return W


def weight_variable_glorot1(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)