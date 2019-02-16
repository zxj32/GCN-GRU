import tensorflow as tf
from gcn_gru.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerRNN(object):
    def __init__(self, model, label_b, label_un, mask):
        self.cost = 0.0
        self.cost_belief = masked_mae_rnn(model.outputs_b, label_b, mask)
        self.cost_uncertain = masked_mae_rnn(model.outputs_u, label_un, mask)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                            tf.square(tf.exp(model.z_log_std)), 1))
        # self.cost -= self.kl
        self.cost = self.cost_belief + self.cost_uncertain
        self.opt_op = self.optimizer.minimize(self.cost)



class OptimizerRNN_ori2(object):
    def __init__(self, model, label_b, label_un, mask):
        self.cost = 0.0
        self.cost_belief = masked_mae_rnn(model.outputs_b, label_b, mask)
        self.cost_belief_s = masked_mse_rnn(model.outputs_b, label_b, mask)
        self.cost_uncertain = masked_mae_rnn(model.outputs_u, label_un, mask)
        self.cost_uncertain_s = masked_mse_rnn(model.outputs_u, label_un, mask)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.cost = self.cost_belief + self.cost_uncertain
        self.opt_op = self.optimizer.minimize(self.cost_belief_s + self.cost_uncertain_s)


class OptimizerRNN_ori(object):
    def __init__(self, model, label_b, label_un, mask):
        self.cost = 0.0
        self.cost_belief = masked_mae_rnn(model.outputs_b, label_b, mask)
        self.cost_belief_s = masked_mse_rnn(model.outputs_b, label_b, mask)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                            tf.square(tf.exp(model.z_log_std)), 1))
        # self.cost -= self.kl
        # self.cost = self.cost_uncertain
        self.cost = self.cost_belief
        self.opt_op = self.optimizer.minimize(self.cost_belief_s )


class OptimizerGCN_ori(object):
    def __init__(self, model, label_b, label_un, mask):
        self.cost = 0.0
        self.cost_belief = masked_mae_rnn(model.belief, label_b, mask)
        self.cost_uncertain = masked_mae_rnn(model.uncertainty, label_un, mask)
        self.cost_belief_s = masked_mae_rnn(model.belief, label_b, mask)
        self.cost_uncertain_s = masked_mse_rnn(model.uncertainty, label_un, mask)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.cost = self.cost_belief_s + self.cost_uncertain_s
        self.opt_op = self.optimizer.minimize(self.cost)
