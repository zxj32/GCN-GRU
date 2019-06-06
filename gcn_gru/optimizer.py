import tensorflow as tf
from gcn_gru.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, label_b, label_un, mask, omega_t):
        preds_sub = preds
        labels_sub = labels

        self.cost = 0.0
        self.cost_belief = masked_mse_abs(model.belief, label_b, mask)
        self.cost_uncertain = masked_mse_abs(model.uncertain, label_un, mask)
        # self.cost_adj = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.cost_decode_sparse = masked_decode_sparse(preds_sub, labels_sub)
        # self.cost_decode = masked_decode(preds_sub, adj_mask)
        self.cost_kl = tf.reduce_mean(model.kl_d)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                            tf.square(tf.exp(model.z_log_std)), 1))
        # self.cost -= self.kl
        self.cost = self.cost_belief + self.cost_uncertain
        self.cost1 = self.cost_belief + self.cost_uncertain + self.cost_decode_sparse * FLAGS.p_encode + self.cost_kl * FLAGS.p_kl
        # self.cost1 = self.cost_belief + self.cost_uncertain + self.cost_kl * FLAGS.p_kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.opt_op1 = self.optimizer.minimize(self.cost1)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
        #                                    tf.cast(labels_sub, tf.int32))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.test_mse = masked_mse_abs(model.belief, label_b, mask) + masked_mse_abs(model.uncertain, label_un, mask)
        self.omega_mse = masked_mse_abs(model.omega, omega_t, mask)


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
