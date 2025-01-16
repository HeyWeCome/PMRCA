import numpy as np
import torch
import pickle
from utils import Constants
import os
from torch_geometric.data import Data
import scipy.sparse as sp
import torch.nn.functional as F

def build_friendship_network(dataloader):
    _u2idx = {}

    with open(dataloader.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    edges_list = []

    if os.path.exists(dataloader.net_data):
        with open(dataloader.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]

            relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _u2idx and edge[1] in _u2idx]
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            edges_list += relation_list_reverse
    else:
        return []

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()

    friend_ship_network = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)
    return friend_ship_network

def build_diff_hyper_graph_list(cascades, timestamps, user_size, step_split=Constants.step_split):
    times, root_list = build_hyper_diff_graph(cascades, timestamps, user_size)

    zero_vec = torch.zeros_like(times)
    one_vec = torch.ones_like(times)

    time_sorted = []
    graph_list = {}

    for time in timestamps:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)

    split_length = len(time_sorted) // step_split

    for x in range(split_length, split_length * step_split, split_length):
        if x == split_length:
            sub_graph = torch.where(times > 0, one_vec, zero_vec) - torch.where(times > time_sorted[x],
                                                                                one_vec,
                                                                                zero_vec)
        else:
            sub_graph = torch.where(times > time_sorted[x - split_length], one_vec, zero_vec) - torch.where(
                times > time_sorted[x], one_vec, zero_vec)

        graph_list[time_sorted[x]] = sub_graph

    graphs = [graph_list, root_list]

    return graphs

def build_hyper_diff_graph(cascades, timestamps, user_size):
    e_size = len(cascades) + 1
    n_size = user_size
    rows = []
    cols = []
    vals_time = []
    root_list = [0]

    for i in range(e_size - 1):
        root_list.append(cascades[i][0])
        rows += cascades[i][:-1]
        cols += [i + 1] * (len(cascades[i]) - 1)
        vals_time += timestamps[i][:-1]

    root_list = torch.tensor(root_list)

    Times = torch.sparse_coo_tensor(torch.Tensor([rows, cols]), torch.Tensor(vals_time), [n_size, e_size])

    return Times.to_dense(), root_list