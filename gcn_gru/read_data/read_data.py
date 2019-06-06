import numpy as np
import random
from scipy import sparse
from collections import Counter


def add_noise(feat_b, feat_u, test_mask, noise, consis_index):
    train_index = []
    conflict_index = []
    noise_num = int(noise * len(feat_b[0]))

    for i in range(len(test_mask)):
        train_index_i = []
        test_mask_i = test_mask[i]
        for j in range(len(test_mask_i)):
            if test_mask_i[j] == False:
                train_index_i.append(j)
        train_index.append(train_index_i)

    for i in range(len(feat_b)):
        consis_index_i = consis_index[i]
        train_index_i = train_index[i]
        consis_train_index_i = [element for element in train_index_i if element in consis_index_i]
        consis_train_index_i = train_index_i
        conflict_index_i = random.sample(consis_train_index_i, noise_num)
        conflict_index.append(conflict_index_i)
        for k in conflict_index_i:
            feat_b[i][k] = 1.0 - feat_b[i][k] - feat_u[i][k]
    return feat_b


def get_mask(num_node, num_test):
    test_index = random.sample(range(num_node), num_test)
    train_mask = np.ones(num_node, dtype=bool)
    test_mask = np.zeros(num_node, dtype=bool)
    for i in test_index:
        test_mask[i] = True
        train_mask[i] = False
    return train_mask, test_mask

def generate_train_test_epinion_noise(test_ratio, node, noise):
    adj_n = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/epinion_node{}_alledges.npy".format(node))
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_{}.npy".format(node))
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/uncertainty_feature_{}.npy".format(node))
    consis_index = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/consis_index_{}.npy".format(node))
    # consis_index = get_consistent_index(b_all, adj_n)
    train_mask = []
    test_mask = []
    num_edge = b_all.shape[1]
    test_num = int(test_ratio * num_edge)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num_edge, test_num)
        train_mask.append(train_mask_i)
        test_mask.append(test_mask_i)
    b_all = add_noise(b_all, u_all, test_mask, noise, consis_index)
    train_feature_b = np.array(b_all)
    train_feature_u = np.array(u_all)
    train_mask = np.reshape(train_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    test_mask = np.reshape(test_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_b = np.reshape(train_feature_b, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_u = np.reshape(train_feature_u, [1, train_feature_u.shape[0], train_feature_u.shape[1]])

    adj = sparse.csr_matrix(adj_n)
    return adj, train_feature_b, train_feature_u, train_mask, test_mask


def generate_train_test_epinion(test_ratio, node):
    adj_n = np.load("read_data/epinion_node{}_alledges.npy".format(node))
    b_all = np.load("read_data/belief_feature_{}.npy".format(node))
    u_all = np.load("read_data/uncertainty_feature_{}.npy".format(node))
    train_mask = []
    test_mask = []
    num_edge = b_all.shape[1]
    test_num = int(test_ratio * num_edge)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num_edge, test_num)
        train_mask.append(train_mask_i)
        test_mask.append(test_mask_i)
    train_feature_b = np.array(b_all)
    train_feature_u = np.array(u_all)
    train_mask = np.reshape(train_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    test_mask = np.reshape(test_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_b = np.reshape(train_feature_b, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_u = np.reshape(train_feature_u, [1, train_feature_u.shape[0], train_feature_u.shape[1]])

    adj = sparse.csr_matrix(adj_n)
    return adj, train_feature_b, train_feature_u, train_mask, test_mask

def generate_train_test_epinion_noise_sparse(test_ratio, node, noise):
    adj_index = np.load("read_data/adj_epinion_{}_index.npy".format(node))
    b_all = np.load("read_data/belief_feature_{}.npy".format(node))
    # b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_{}_noise.npy".format(node))
    u_all = np.load("read_data/uncertainty_feature_{}.npy".format(node))
    # consis_index = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/consis_index_{}.npy".format(node))
    train_mask = []
    test_mask = []
    num_edge = b_all.shape[1]
    test_num = int(test_ratio * num_edge)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num_edge, test_num)
        train_mask.append(train_mask_i)
        test_mask.append(test_mask_i)
    #b_all = add_noise(b_all, u_all, test_mask, noise, consis_index)
    #b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_47676_noise.npy")
    train_feature_b = np.array(b_all)
    train_feature_u = np.array(u_all)
    train_mask = np.reshape(train_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    test_mask = np.reshape(test_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_b = np.reshape(train_feature_b, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_u = np.reshape(train_feature_u, [1, train_feature_u.shape[0], train_feature_u.shape[1]])
    row = adj_index[:, 0]
    col = adj_index[:, 1]
    print(np.max(row), np.max(col))
    value = np.ones(len(adj_index))
    adj = sparse.coo_matrix((value, (row, col)), shape=(num_edge, num_edge))
    return adj, train_feature_b, train_feature_u, train_mask, test_mask

def generate_train_test_epinion_sparse(test_ratio, node):
    adj_index = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/adj_epinion_{}_index.npy".format(node))
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_{}.npy".format(node))
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/uncertainty_feature_{}.npy".format(node))
    train_mask = []
    test_mask = []
    num_edge = b_all.shape[1]
    test_num = int(test_ratio * num_edge)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num_edge, test_num)
        train_mask.append(train_mask_i)
        test_mask.append(test_mask_i)
    train_feature_b = np.array(b_all)
    train_feature_u = np.array(u_all)
    train_mask = np.reshape(train_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    test_mask = np.reshape(test_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_b = np.reshape(train_feature_b, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_u = np.reshape(train_feature_u, [1, train_feature_u.shape[0], train_feature_u.shape[1]])
    row = adj_index[:, 0]
    col = adj_index[:, 1]
    print(np.max(row), np.max(col))
    value = np.ones(len(adj_index))
    adj = sparse.coo_matrix((value, (row, col)), shape=(num_edge, num_edge))
    return adj, train_feature_b, train_feature_u, train_mask, test_mask

