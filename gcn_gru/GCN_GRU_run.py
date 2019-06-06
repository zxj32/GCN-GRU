from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gcn_gru.optimizer import OptimizerAE, OptimizerVAE, OptimizerRNN
from gcn_gru.model import GCNModelAE, GCNModelVAE, GCNModelRNN, GCNModelRNN_ori
from gcn_gru.preprocessing import preprocess_graph, construct_feed_dict_rnn, sparse_to_tuple, mask_test_edge_opinion
from gcn_gru.read_data.read_data import generate_train_test_epinion

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden_units', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('test_ratio', 0.2, 'test number of dataset.')
flags.DEFINE_float('noise', 0.1, 'noise of synthetic data.')
flags.DEFINE_integer('bias', 0, 'bias.')
flags.DEFINE_integer('node', 500, 'node size.')

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

print("seed:", seed, "hidden_units:", FLAGS.hidden_units, "dropout:", FLAGS.dropout, "test_ratio:", FLAGS.test_ratio, "noise:", FLAGS.noise, "bias:", FLAGS.bias, "node:", FLAGS.node)

# adj, label_b, label_u, train_mask, test_mask = generate_train_test_epinion_noise(FLAGS.test_ratio, FLAGS.node, FLAGS.noise)
adj, label_b, label_u, train_mask, test_mask = generate_train_test_epinion(FLAGS.test_ratio, FLAGS.node)

features = []
if FLAGS.features == 0:
    # features = sp.identity(y_test_belief.shape[0])  # featureless
    for i in range(label_b.shape[1]):
        features.append(np.identity(adj.shape[0]))
    features = np.reshape(features, [1, label_b.shape[1], adj.shape[0] * adj.shape[0]])

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(1, label_b.shape[1], adj.shape[0] * adj.shape[0])),
    'adj': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'labels_b': tf.placeholder(tf.float32, shape=(1, label_b.shape[1], label_b.shape[2])),
    'labels_un': tf.placeholder(tf.float32, shape=(1, label_b.shape[1], label_b.shape[2])),
    'labels_mask': tf.placeholder(tf.int32)
}

num_nodes = adj.shape[0]

# Create model

model = GCNModelRNN(placeholders, num_nodes, bias=FLAGS.bias)

# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerRNN(model=model, label_b=placeholders['labels_b'], label_un=placeholders['labels_un'],
                       mask=placeholders['labels_mask'])

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess = tf.Session()

for i in range(1):
    # seed = i
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    # random.seed(seed)
    adj, label_b, label_u, train_mask, test_mask = generate_train_test_epinion(FLAGS.test_ratio, FLAGS.node)
    # Train model
    sess.run(tf.global_variables_initializer())
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict_rnn(adj_norm, features, placeholders, label_b, label_u, train_mask)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([opt.opt_op, opt.cost, model.outputs_b, model.outputs_u], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "time=",
              "{:.5f}".format(time.time() - t))
    print("Optimization Finished!")

    feed_dict = construct_feed_dict_rnn(adj_norm, features, placeholders, label_b, label_u, test_mask)
    outs = sess.run([opt.cost, opt.cost, opt.cost_belief, opt.cost_uncertain], feed_dict=feed_dict)
    test_cost = outs[1]
    print("test cost =", "{:.3f}".format(outs[1]), "belief mae =", "{:.3f}".format(outs[2]), "uncertainty mae =", "{:.3f}".format(outs[3]))
