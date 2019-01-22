import numpy as np
import random
from scipy import sparse
from collections import Counter


def get_neighbor(adj):
    neigh = []
    for item in adj:
        neigh_i = []
        print(1)
        for i in range(np.size(item)):
            if item[i] == 1.0:
                neigh_i.append(i)
        neigh.append(neigh_i)
    return neigh


def get_omega_obs(obs):
    W = 2.0
    r = Counter(obs)[1]
    s = Counter(obs)[0]
    u = W / (W + r + s)
    b = r / (W + r + s)
    d = s / (W + r + s)
    return [b, d, u, r, s]

def get_mask(num_node, num_test):
    test_index = random.sample(range(num_node), num_test)
    train_mask = np.ones(num_node, dtype=bool)
    test_mask = np.zeros(num_node, dtype=bool)
    for i in test_index:
        test_mask[i] = True
        train_mask[i] = False
    return train_mask, test_mask


def get_b_u(weekday):
    bb = []
    uu = []
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/dc_belief_T38_0.8.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/dc_uncertain_T38_0.8.npy")
    b_w = b_all[weekday * 36:(weekday + 1) * 36]
    u_w = u_all[weekday * 36:(weekday + 1) * 36]
    for i in range(len(b_w)):
        if np.mod(i, 6) == 0:
            bb.append(b_w[i])
            uu.append(u_w[i])
    return bb, uu


def get_b_u_6_21(weekday):
    bb = []
    uu = []
    b_all = np.load("./data/belief_feature_ref0.8.npy")
    u_all = np.load("./data/uncertainty_feature_ref0.8.npy")
    b_w = b_all[weekday * 96:(weekday + 1) * 96]
    u_w = u_all[weekday * 96:(weekday + 1) * 96]
    for i in range(len(b_w)):
        if np.mod(i, 6) == 0:
            bb.append(b_w[i])
            uu.append(u_w[i])
    return bb, uu


def get_b_u_6_21_pa(weekday):
    bb = []
    uu = []
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/traffic_data/PA_data/feature/belief_feature_ref0.8.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/traffic_data/PA_data/feature/uncertainty_feature_ref0.8.npy")
    b_w = b_all[weekday * 96:(weekday + 1) * 96]
    u_w = u_all[weekday * 96:(weekday + 1) * 96]
    for i in range(len(b_w)):
        if np.mod(i, 6) == 0:
            bb.append(b_w[i])
            uu.append(u_w[i])
    return bb, uu

def get_consistent_index(feat_b, adj):
    consistent_index = []
    neighbors = get_neighbor(adj)
    for i in range(len(feat_b)):
        feat_i = feat_b[i]
        consistent_index_i = []
        for j in range(len(feat_i)):
            neigh = neighbors[j]
            ave_b = np.sum((adj[j] * feat_i)) / len(neigh)
            if np.abs(feat_i[j] - ave_b) < 0.1:  # parameter of define conflict
                consistent_index_i.append(j)
        consistent_index.append(consistent_index_i)
        # print(len(consistent_index_i))
    return consistent_index


def get_consistent_index_sparse(feat_b, neigh_):
    consistent_index = []
    neighbors = neigh_
    for i in range(len(feat_b)):
        feat_i = feat_b[i]
        consistent_index_i = []
        for j in range(len(feat_i)):
            neigh = neighbors[j]
            ave_b = 0
            if neigh == []:
                print("none")
            else:
                for k in neigh:
                    ave_b += feat_i[k]
                ave_b = ave_b / len(neigh)
            if np.abs(feat_i[j] - ave_b) < 0.1:  # parameter of define conflict
                consistent_index_i.append(j)
        consistent_index.append(consistent_index_i)
        # print(len(consistent_index_i))
    return consistent_index


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

def add_noise_spam(feat_b, feat_u, test_mask, noise, consis_index):
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
        # consis_index_i = consis_index[i]
        train_index_i = train_index[i]
        # consis_train_index_i = [element for element in train_index_i if element in consis_index_i]
        consis_train_index_i = train_index_i
        conflict_index_i = random.sample(consis_train_index_i, noise_num)
        conflict_index.append(conflict_index_i)
        for k in conflict_index_i:
            feat_b[i][k] = 1.0 - feat_b[i][k] - feat_u[i][k]
    return feat_b


def generate_train_test_dc_noise(test_ratio, weekday, noise):
    adj_n = np.load("./data/adjacency_matrix_dc_milcom.npy")
    b_all, u_all = get_b_u_6_21(weekday)
    consis_index = get_consistent_index(b_all, adj_n)
    train_mask = []
    test_mask = []
    num = len(adj_n)
    test_num = int(test_ratio * num)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num, test_num)
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


def generate_train_test_epinion_noise_sparse(test_ratio, node, noise):
    adj_index = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/adj_epinion_{}_index.npy".format(node))
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_{}.npy".format(node))
    # b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_{}_noise.npy".format(node))
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/uncertainty_feature_{}.npy".format(node))
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

def generate_train_test_spam_noise(test_ratio, node, noise):
    num_node = 5607448
    # adj_n = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/epinion_node{}_alledges.npy".format(node))
    # b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_{}.npy".format(node))
    # u_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/uncertainty_feature_{}.npy".format(node))
    b_all = np.random.normal(size=[10, num_node])
    u_all = np.random.normal(size=[10, num_node])
    consis_index = [1, 2, 3, 4, 56, 7]
    adj_index = np.load("/network/rit/lab/ceashpc/xujiang/spammer_data/adj_all_index.npy")
    row = adj_index[:, 0]-1
    col = adj_index[:, 1]-1
    print(np.max(row), np.max(col))
    value = np.ones(len(adj_index))
    adj = sparse.coo_matrix((value, (row, col)), shape=(num_node, num_node))
    train_mask = []
    test_mask = []
    test_num = int(test_ratio * num_node)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num_node, test_num)
        train_mask.append(train_mask_i)
        test_mask.append(test_mask_i)
    # b_all = add_noise(b_all, u_all, test_mask, noise, consis_index)
    train_feature_b = np.array(b_all)
    train_feature_u = np.array(u_all)
    train_mask = np.reshape(train_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    test_mask = np.reshape(test_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_b = np.reshape(train_feature_b, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_u = np.reshape(train_feature_u, [1, train_feature_u.shape[0], train_feature_u.shape[1]])

    # adj = sparse.csr_matrix(adj_n)
    return adj, train_feature_b, train_feature_u, train_mask, test_mask


def generate_train_test_spam_noise_sample(test_ratio, node, noise):
    b_all = np.load('/network/rit/lab/ceashpc/xujiang/project/Spammer/output/Obs/4hours/4_hours_graph_belief_sample_' + str(node) + '.npy')
    u_all = np.load('/network/rit/lab/ceashpc/xujiang/project/Spammer/output/Obs/4hours/4_hours_graph_uncertainty_sample_' + str(
        node) + '.npy')
    num_node = b_all.shape[1]
    consis_index = []
    adj_index = np.load('/network/rit/lab/ceashpc/xujiang/spammer_data/sample_nodes/sample_undirect_index_map_' + str(node) + '.npy')
    row = adj_index[:, 0]
    col = adj_index[:, 1]
    value = np.ones(len(adj_index))
    adj = sparse.coo_matrix((value, (row, col)), shape=(num_node, num_node))
    train_mask = []
    test_mask = []
    test_num = int(test_ratio * num_node)
    for i in range(len(b_all)):
        train_mask_i, test_mask_i = get_mask(num_node, test_num)
        train_mask.append(train_mask_i)
        test_mask.append(test_mask_i)
    b_all = add_noise_spam(b_all, u_all, test_mask, noise, consis_index)
    train_feature_b = np.array(b_all)
    train_feature_u = np.array(u_all)
    train_mask = np.reshape(train_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    test_mask = np.reshape(test_mask, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_b = np.reshape(train_feature_b, [1, train_feature_b.shape[0], train_feature_b.shape[1]])
    train_feature_u = np.reshape(train_feature_u, [1, train_feature_u.shape[0], train_feature_u.shape[1]])

    # adj = sparse.csr_matrix(adj_n)
    return adj, train_feature_b, train_feature_u, train_mask, test_mask




def get_consistant():
    # adj_n = np.load(
        # "/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/epinion_node500_alledges.npy")
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/belief_feature_500.npy")
    # u_all = np.load("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/uncertainty_feature_500.npy")
    neigh = np.load("/network/rit/lab/ceashpc/xujiang/project/GCN_RNN/gae/traffic_data/nodes-500_neigh.npy")
    consis_index = get_consistent_index_sparse(b_all, neigh)
    np.save("/network/rit/lab/ceashpc/xujiang/eopinion_data/feature/consis_index_500.npy", consis_index)

