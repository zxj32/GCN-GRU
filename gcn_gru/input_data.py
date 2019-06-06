import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def load_data_epinion():
    adj = np.load("./traffic_data/epinion_node1000_alledges.npy")
    # adj_mask = np.load("./data/adjacency_matrix_dc_milcom.npy")
    features = sp.identity(len(adj))
    # adj_mask = np.reshape(adj_mask, [-1])
    adj = sp.csr_matrix(adj)
    return adj, features

def load_data_1():
    adj = np.load("./data/adjacency_matrix_ph_milcom.npy")
    # adj_mask = np.load("./data/adjacency_matrix_dc_milcom.npy")
    features = sp.identity(len(adj))
    # adj_mask = np.reshape(adj_mask, [-1])
    adj = sp.csr_matrix(adj)
    return adj, features


def load_data_dc():
    adj = np.load("./data/adjacency_matrix_dc_milcom.npy")
    # adj_mask = np.load("./data/adjacency_matrix_dc_milcom.npy")
    features = sp.identity(len(adj))
    # adj_mask = np.reshape(adj_mask, [-1])
    adj = sp.csr_matrix(adj)
    return adj, features

def load_data_before():
    adj = np.load("./data/adjacency_matrix_dc_milcom.npy")
    # adj_mask = np.load("./data/adjacency_matrix_dc_milcom.npy")
    features = sp.identity(len(adj))
    # adj_mask = np.reshape(adj_mask, [-1])
    adj = sp.csr_matrix(adj)
    return adj, features


def load_data_beijing():
    adj = sp.load_npz("./traffic_data/adj_undirect_beijing_sparse.npz")
    # adj = np.load("./traffic_data/adj_undirect_beijing.npy")
    # adj_mask = np.load("./traffic_data/adj_undirect_beijing.npy")
    # adj_mask = np.reshape(adj_mask, [-1])
    # adj_mask = sp.csr_matrix(adj_mask)

    features = sp.identity(adj.shape[0])
    # adj = sp.csr_matrix(adj)
    return adj, features
