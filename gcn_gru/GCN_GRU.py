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
from gcn_gru.model import GCNModelAE, GCNModelVAE, GCNModelRNN, GCNModelRNN_ori, GCNModelRNN_sparse
from gcn_gru.preprocessing import preprocess_graph, construct_feed_dict_rnn, sparse_to_tuple, mask_test_edge_opinion, construct_feed_dict_rnn_sparse
from gcn_gru.traffic_data.read_dc import generate_train_test_dc_noise, generate_train_test_pa_noise

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden_units', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('test_ratio', 0.1, 'test number of dataset.')
flags.DEFINE_float('noise', 0.1, 'noise of synthetic data.')
flags.DEFINE_integer('bias', 0, 'bias.')

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

print("seed:", seed, "hidden_units:", FLAGS.hidden_units, "dropout:", FLAGS.dropout, "test_ratio:", FLAGS.test_ratio, "noise:", FLAGS.noise, "bias:", FLAGS.bias)

adj, label_b, label_u, train_mask, test_mask = generate_train_test_dc_noise(FLAGS.test_ratio, 0, FLAGS.noise)

features = []
if FLAGS.features == 0:
    features = np.ones([1, label_b.shape[1], 10])
indices = [[x, x] for x in range(adj.shape[0])]
values = np.ones(adj.shape[0])
# Some preprocessing
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
#sess = tf.Session(config=config)
sess = tf.Session()

ave_cost = []
ave_b = []
ave_u = []
for i in range(5):
    # seed = i
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    # random.seed(seed)
    adj, label_b, label_u, train_mask, test_mask = generate_train_test_dc_noise(FLAGS.test_ratio, i, FLAGS.noise)
    # Train model
    sess.run(tf.global_variables_initializer())
    best_cost = 100.0
    best_belief = 100.0
    best_uncertainty = 100.0
    best_epoch = 0
    feed_dict = construct_feed_dict_rnn_sparse(adj_norm, features, placeholders, label_b, label_u, train_mask, indices,
                                               values)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    t1 = time.time()
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        # feed_dict = construct_feed_dict_rnn(adj_norm, features, placeholders, label_b, label_u, train_mask)
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, model.outputs_b, model.outputs_u], feed_dict=feed_dict)
        # Compute average loss
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "time=",
              "{:.5f}".format(time.time() - t))
        if np.mod(epoch + 1, 100) == 0:
            #feed_dict = construct_feed_dict_rnn(adj_norm, features, placeholders, label_b, label_u, test_mask)
            feed_dict = construct_feed_dict_rnn_sparse(adj_norm, features, placeholders, label_b, label_u, test_mask,
                                                       indices, values)

            # Run single weight update
            outs = sess.run([opt.cost, opt.cost, opt.cost_belief, opt.cost_uncertain], feed_dict=feed_dict)
            # Compute test loss
            test_cost = outs[1]
            print("epoch : ", epoch + 1, "test all mae =", "{:.3f}".format(outs[1]), "belief mae =", "{:.3f}".format(outs[2]), "uncertainty mae =", "{:.3f}".format(outs[3]))
            if test_cost < best_cost:
                best_cost = test_cost
                best_belief = outs[2]
                best_uncertainty = outs[3]
                best_epoch = epoch
    # print("Optimization Finished!")
    print(time.time()-t1)
    print("best epoch:", best_epoch, "best cost:", best_cost)
    ave_cost.append(best_cost)
    ave_b.append(best_belief)
    ave_u.append(best_uncertainty)
print("average cost:", np.mean(ave_cost), "average belief:", np.mean(ave_b), "average uncertainty:", np.mean(ave_u))
