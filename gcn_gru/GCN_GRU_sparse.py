from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gcn_gru.optimizer import OptimizerAE, OptimizerVAE, OptimizerRNN
from gcn_gru.input_data import load_data, load_data_1
from gcn_gru.model import GCNModelAE, GCNModelVAE, GCNModelRNN, GCNModelRNN_ori, GCNModelRNN_sparse
from gcn_gru.preprocessing import preprocess_graph, construct_feed_dict_rnn, sparse_to_tuple, construct_feed_dict_rnn_sparse
from gcn_gru.traffic_data.read_dc import generate_train_test_epinion_noise, generate_train_test_epinion_noise_sparse

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden_units', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('test_ratio', 0.2, 'test number of dataset.')
flags.DEFINE_float('noise', 0.1, 'noise of synthetic data.')
flags.DEFINE_integer('bias', 0, 'bias.')
flags.DEFINE_integer('node', 5000, 'node size.')

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

print("seed:", seed, "hidden_units:", FLAGS.hidden_units, "dropout:", FLAGS.dropout, "learning rate:", FLAGS.learning_rate, "test_ratio:", FLAGS.test_ratio, "noise:", FLAGS.noise, "bias:", FLAGS.bias, "node:", FLAGS.node)

adj, label_b, label_u, train_mask, test_mask = generate_train_test_epinion_noise_sparse(FLAGS.test_ratio, FLAGS.node, FLAGS.noise)

features = []
if FLAGS.features == 0:
    # features = sp.identity(y_test_belief.shape[0])  # featureless
    # for i in range(label_b.shape[1]):
    #     features.append(np.identity(adj.shape[0]))
    features = np.ones([1, label_b.shape[1], 10])

indices = [[x, x] for x in range(adj.shape[0])]
values = np.ones(adj.shape[0])

adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(1, label_b.shape[1], 10)),
    # 'features_rnn': tf.sparse_placeholder(tf.float32),
    'index': tf.placeholder(tf.int64, shape=(adj.shape[0], 2)),
    'value': tf.placeholder(tf.float32, shape=(adj.shape[0])),
    'adj': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'labels_b': tf.placeholder(tf.float32, shape=(1, label_b.shape[1], label_b.shape[2])),
    'labels_un': tf.placeholder(tf.float32, shape=(1, label_b.shape[1], label_b.shape[2])),
    'labels_mask': tf.placeholder(tf.int32)
}

num_nodes = adj.shape[0]

# Create model

model = GCNModelRNN_sparse(placeholders, num_nodes, bias=FLAGS.bias)

# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerRNN(model=model, label_b=placeholders['labels_b'], label_un=placeholders['labels_un'],
                       mask=placeholders['labels_mask'])

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session()

for i in range(1):
    # seed = i
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    # random.seed(seed)
    # Train model
    sess.run(tf.global_variables_initializer())
    feed_dict = construct_feed_dict_rnn_sparse(adj_norm, features, placeholders, label_b, label_u, train_mask, indices,
                                               values)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    t1 = time.time()
    for epoch in range(FLAGS.epochs):
        t = time.time()
        outs = sess.run([opt.opt_op, opt.cost, model.outputs_b, model.outputs_u, opt.cost_belief, opt.cost_uncertain], feed_dict=feed_dict)
        # Compute  loss
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "belief mae =", "{:.3f}".format(outs[4]), "uncertainty mae =", "{:.3f}".format(outs[5]), "time=",
             "{:.5f}".format(time.time() - t))
    print("Optimization Finished!")

    feed_dict = construct_feed_dict_rnn(adj_norm, features, placeholders, label_b, label_u, test_mask)
    outs = sess.run([opt.cost, opt.cost, opt.cost_belief, opt.cost_uncertain], feed_dict=feed_dict)
    test_cost = outs[1]
    print("test cost =", "{:.3f}".format(outs[1]), "belief mae =", "{:.3f}".format(outs[2]), "uncertainty mae =", "{:.3f}".format(outs[3]))