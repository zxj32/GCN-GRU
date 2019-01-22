from gcn_gru.layers import GraphConvolution, InnerProductDecoder, InnerProductDecoder_Opinion, GraphConvolutionSparse, \
    GraphConvolution_saprse
import tensorflow as tf
from gcn_gru.GRU_cell import GCNGRUCell, GRUCell_ori, GCNGRUCell_sparse

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.label_b = placeholders['labels_b']
        self.label_un = placeholders['labels_un']
        self.mask = placeholders['labels_mask']
        self.alpha_0 = placeholders['alpha_0']
        self.beta_0 = placeholders['beta_0']
        self.m = FLAGS.KL_m
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.belief = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=tf.nn.sigmoid,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.uncertain = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=tf.nn.sigmoid,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)
        # decode define
        self.b = tf.minimum(self.belief + 0.0000001, 0.999999)
        self.u = tf.minimum(self.uncertain + 0.0000001, 0.999999)
        self.W = 2.0
        self.r = self.W * self.b / self.u + 0.0000001
        self.d = tf.minimum(tf.nn.relu(1 - self.b - self.u) + 0.0000001, 0.999999)
        self.s = self.W * self.d / self.u + 0.0000001
        self.a = 0.5
        self.alpha = self.r + self.W * self.a + 0.0000001
        self.beta = self.s + self.W * (1 - self.a) + 0.0000001
        self.uni = tf.random_uniform([self.n_samples, FLAGS.hidden2], 0.001, 0.999)
        self.z_n_b = tf.minimum((1.0 - tf.pow(self.uni, tf.reciprocal(self.beta)) + 0.0000001), 0.999999)
        self.z_n = tf.minimum(tf.pow(self.z_n_b, tf.reciprocal(self.alpha)) + 0.0000001, 0.999999)
        self.z = tf.distributions.Normal(loc=0.0, scale=1.0).quantile(self.z_n)
        # self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
        self.reconstructions = InnerProductDecoder_Opinion(input_dim=FLAGS.hidden2,
                                                           act=tf.nn.sigmoid,
                                                           logging=self.logging)(self.z)
        # KL-divergence
        self.digmma_beta = tf.digamma(self.beta)
        self.kl_1 = (1.0 - self.alpha_0 / self.alpha) * (
                -0.5772 - self.digmma_beta - tf.reciprocal(self.beta)) + tf.log(self.alpha * self.beta) + tf.lbeta(
            [self.alpha_0, self.beta_0]) - 1.0 + tf.reciprocal(self.beta)
        self.kl_2 = 0.0
        for m in range(1, self.m + 1):
            self.kl_2 += tf.reciprocal(m + self.alpha * self.beta) * tf.exp(
                tf.lbeta(tf.stack([m / self.alpha, self.beta], axis=2)))
        self.kl_d = self.kl_1 + self.beta * (self.beta_0 - 1.0) * self.kl_2
        # omege
        self.omega = self.alpha / (self.alpha + self.beta)


class GCNModel_ori(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModel_ori, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolution(input_dim=self.n_samples,
                                        output_dim=10,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.belief = GraphConvolution(input_dim=10,
                                       output_dim=1,
                                       adj=self.adj,
                                       act=tf.nn.sigmoid,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.hidden2 = GraphConvolution(input_dim=self.n_samples,
                                        output_dim=10,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.uncertainty = GraphConvolution(input_dim=10,
                                            output_dim=1,
                                            adj=self.adj,
                                            act=tf.nn.sigmoid,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.hidden2)


class GCNModel_ori3(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModel_ori3, self).__init__(**kwargs)

        self.index = placeholders['index']
        self.value = placeholders['value']
        self.inputs = tf.SparseTensor(self.index, self.value, [num_nodes, num_nodes])
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.build()

    def _build(self):
        non_zeros_feat = tf.size(self.inputs.values)
        self.hidden1 = GraphConvolution_saprse(input_dim=self.n_samples,
                                               output_dim=10,
                                               adj=self.adj,
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               non_zero=non_zeros_feat,
                                               logging=self.logging)(self.inputs)

        self.belief = GraphConvolution(input_dim=10,
                                       output_dim=1,
                                       adj=self.adj,
                                       act=tf.nn.sigmoid,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.hidden2 = GraphConvolution_saprse(input_dim=self.n_samples,
                                               output_dim=10,
                                               adj=self.adj,
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               non_zero=non_zeros_feat,
                                               logging=self.logging)(self.inputs)

        self.uncertainty = GraphConvolution(input_dim=10,
                                            output_dim=1,
                                            adj=self.adj,
                                            act=tf.nn.sigmoid,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.hidden2)


class GCNModel_ori2(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModel_ori2, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.inouts_u = placeholders['features_u']
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolution(input_dim=1,
                                        output_dim=10,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.belief = GraphConvolution(input_dim=10,
                                       output_dim=1,
                                       adj=self.adj,
                                       act=tf.nn.sigmoid,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.hidden2 = GraphConvolution(input_dim=1,
                                        output_dim=10,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inouts_u)

        self.uncertainty = GraphConvolution(input_dim=10,
                                            output_dim=1,
                                            adj=self.adj,
                                            act=tf.nn.sigmoid,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.hidden2)


class GCNModel_ori_spam(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModel_ori_spam, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.inouts_u = placeholders['features_u']
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.build()

    def _build(self):
        self.belief = GraphConvolution(input_dim=1,
                                       output_dim=1,
                                       adj=self.adj,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.inputs)

        self.uncertainty = GraphConvolution(input_dim=1,
                                            output_dim=1,
                                            adj=self.adj,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.inouts_u)


class GCNModelRNN_sparse(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModelRNN_sparse, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        # self.input_rnn = placeholders['features_rnn']
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.index = placeholders['index']
        self.value = placeholders['value']
        self.build()

    def _build(self):
        self.cell_1 = GCNGRUCell_sparse(num_units=self.hidden_units, adj_mx=self.adj, num_nodes=self.n_samples,
                                        dropout=self.dropout, bias=self.bias, index=self.index, value=self.value)
        self.cell_2 = GCNGRUCell_sparse(num_units=1, adj_mx=self.adj, num_nodes=self.n_samples, dropout=self.dropout,
                                        bias=self.bias, index=self.index, value=self.value)
        self.cell_3 = GCNGRUCell_sparse(num_units=1, adj_mx=self.adj, num_nodes=self.n_samples, dropout=self.dropout,
                                        bias=self.bias, index=self.index, value=self.value)
        self.cell_b = [self.cell_1, self.cell_2]
        self.cell_u = [self.cell_1, self.cell_3]
        self.cell_multi_b = tf.nn.rnn_cell.MultiRNNCell(self.cell_b, state_is_tuple=True)
        self.cell_multi_u = tf.nn.rnn_cell.MultiRNNCell(self.cell_u, state_is_tuple=True)

        self.initial_state = self.cell_multi_b.zero_state(1, tf.float32)

        self.outputs_b, _ = tf.nn.dynamic_rnn(self.cell_multi_b, self.inputs, initial_state=self.initial_state)
        self.outputs_u, _ = tf.nn.dynamic_rnn(self.cell_multi_u, self.inputs, initial_state=self.initial_state)


class GCNModelRNN(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModelRNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.build()

    def _build(self):
        self.cell_1 = GCNGRUCell(num_units=self.hidden_units, adj_mx=self.adj, num_nodes=self.n_samples,
                                 dropout=self.dropout, bias=self.bias)
        self.cell_2 = GCNGRUCell(num_units=1, adj_mx=self.adj, num_nodes=self.n_samples, dropout=self.dropout,
                                 bias=self.bias)
        self.cell_3 = GCNGRUCell(num_units=1, adj_mx=self.adj, num_nodes=self.n_samples, dropout=self.dropout,
                                 bias=self.bias)
        self.cell_b = [self.cell_1, self.cell_2]
        self.cell_u = [self.cell_1, self.cell_3]
        self.cell_multi_b = tf.nn.rnn_cell.MultiRNNCell(self.cell_b, state_is_tuple=True)
        self.cell_multi_u = tf.nn.rnn_cell.MultiRNNCell(self.cell_u, state_is_tuple=True)

        self.initial_state = self.cell_multi_b.zero_state(1, tf.float32)

        self.outputs_b, _ = tf.nn.dynamic_rnn(self.cell_multi_b, self.inputs, initial_state=self.initial_state)
        self.outputs_u, _ = tf.nn.dynamic_rnn(self.cell_multi_u, self.inputs, initial_state=self.initial_state)


class GCNModelRNN_ori(Model):
    def __init__(self, placeholders, num_nodes, bias, **kwargs):
        super(GCNModelRNN_ori, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.n_samples = num_nodes
        self.hidden_units = FLAGS.hidden_units
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.bias = bool(bias)
        self.build()

    def _build(self):
        self.cell_1 = GRUCell_ori(num_units=self.hidden_units, adj_mx=self.adj, num_nodes=self.n_samples,
                                  dropout=self.dropout, bias=self.bias)
        self.cell_4 = GRUCell_ori(num_units=self.hidden_units, adj_mx=self.adj, num_nodes=self.n_samples,
                                  dropout=self.dropout, bias=self.bias)
        self.cell_2 = GRUCell_ori(num_units=1, adj_mx=self.adj, num_nodes=self.n_samples, dropout=self.dropout,
                                  bias=self.bias)
        self.cell_3 = GRUCell_ori(num_units=1, adj_mx=self.adj, num_nodes=self.n_samples, dropout=self.dropout,
                                  bias=self.bias)
        self.cell_b = [self.cell_1, self.cell_2]
        self.cell_u = [self.cell_4, self.cell_3]
        self.cell_multi_b = tf.nn.rnn_cell.MultiRNNCell(self.cell_b, state_is_tuple=True)
        self.cell_multi_u = tf.nn.rnn_cell.MultiRNNCell(self.cell_u, state_is_tuple=True)

        self.initial_state = self.cell_multi_b.zero_state(1, tf.float32)

        self.outputs_b, _ = tf.nn.dynamic_rnn(self.cell_multi_b, self.inputs, initial_state=self.initial_state)
        self.outputs_u, _ = tf.nn.dynamic_rnn(self.cell_multi_u, self.inputs, initial_state=self.initial_state)


class GCNModelRNN_ori1(Model):
    def __init__(self, placeholders, num_nodes, **kwargs):
        super(GCNModelRNN_ori1, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        # self.cell_1 = tf.nn.rnn_cell.GRUCell(num_units=16)
        # self.cell_2 = tf.nn.rnn_cell.GRUCell(num_units=1)
        # self.cell_3 = tf.nn.rnn_cell.GRUCell(num_units=1)
        # self.cell_b = [self.cell_1, self. cell_2]
        # self.cell_u = [self.cell_1, self.cell_3]
        # with tf.variable_scope("belief"):
        # self.cell_b = [tf.nn.rnn_cell.GRUCell(size) for size in [self.n_samples]]
        self.cell_b = tf.nn.rnn_cell.GRUCell(self.n_samples)
        # with tf.variable_scope("uncertainty"):
        # self.cell_u = [tf.nn.rnn_cell.GRUCell(size) for size in [10, 1]]
        self.cell_multi_b = tf.nn.rnn_cell.MultiRNNCell(self.cell_b, state_is_tuple=True)
        # self.cell_multi_u = tf.nn.rnn_cell.MultiRNNCell(self.cell_u, state_is_tuple=True)

        self.initial_state = self.cell_multi_b.zero_state(1, tf.float32)

        self.outputs_b, _ = tf.nn.dynamic_rnn(self.cell_multi_b, self.inputs, initial_state=self.initial_state)
        # self.outputs_u, _ = tf.nn.dynamic_rnn(self.cell_multi_u, self.inputs, initial_state=self.initial_state)


class GCNModelRNN_ori2(Model):
    def __init__(self, placeholders, num_nodes, **kwargs):
        super(GCNModelRNN_ori2, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        # self.cell_1 = tf.nn.rnn_cell.GRUCell(num_units=16)
        # self.cell_2 = tf.nn.rnn_cell.GRUCell(num_units=1)
        # self.cell_3 = tf.nn.rnn_cell.GRUCell(num_units=1)
        # self.cell_b = [self.cell_1, self. cell_2]
        # self.cell_u = [self.cell_1, self.cell_3]
        # with tf.variable_scope("belief"):
        # self.cell_b = [tf.nn.rnn_cell.GRUCell(size) for size in [self.n_samples]]
        self.cell_b = tf.nn.rnn_cell.GRUCell(self.n_samples)
        # with tf.variable_scope("uncertainty"):
        # self.cell_u = [tf.nn.rnn_cell.GRUCell(size) for size in [10, 1]]
        # self.cell_multi_u = tf.nn.rnn_cell.MultiRNNCell(self.cell_u, state_is_tuple=True)

        self.initial_state = self.cell_b.zero_state(1, tf.float32)

        self.outputs_b, _ = tf.nn.dynamic_rnn(self.cell_b, self.inputs, initial_state=self.initial_state)
        # self.outputs_u, _ = tf.nn.dynamic_rnn(self.cell_multi_u, self.inputs, initial_state=self.initial_state)
