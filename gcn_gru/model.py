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

