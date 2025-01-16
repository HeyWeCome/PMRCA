import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter

class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class GraphNN(nn.Module):
    def __init__(self, num_nodes, input_dim, dropout=0.5, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, input_dim, padding_idx=0)
        self.gnn1 = GCNConv(input_dim, input_dim * 2)
        self.gnn2 = GCNConv(input_dim * 2, input_dim)
        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        return graph_output.cuda()

class HierarchicalGNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, use_norm=True):
        super(HierarchicalGNNWithAttention, self).__init__()
        self.dropout = dropout
        self.use_norm = use_norm
        if self.use_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)
        self.gat1 = HGATLayer(input_dim, output_dim, dropout=self.dropout, transfer=False, concat=True, use_edge=True)
        self.fusion_layer = Fusion(output_dim)

    def forward(self, friendship_embedding, hypergraph_structure):
        root_embedding = F.embedding(hypergraph_structure[1].cuda(), friendship_embedding)
        hypergraph = hypergraph_structure[0]
        embedding_results = {}

        for subgraph_key in hypergraph.keys():
            subgraph = hypergraph[subgraph_key]
            sub_node_embedding, sub_edge_embedding = self.gat1(friendship_embedding, subgraph.cuda(), root_embedding)
            sub_node_embedding = F.dropout(sub_node_embedding, self.dropout, training=self.training)
            if self.use_norm:
                sub_node_embedding = self.batch_norm1(sub_node_embedding)
                sub_edge_embedding = self.batch_norm1(sub_edge_embedding)
            friendship_embedding = self.fusion_layer(friendship_embedding, sub_node_embedding)
            embedding_results[subgraph_key] = [friendship_embedding.cpu(), sub_edge_embedding.cpu()]

        return embedding_results

class HGATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, transfer, concat=True, bias=False, use_edge=True):
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat = concat
        self.use_edge = use_edge
        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.input_dim, self.output_dim))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.weight3 = Parameter(torch.Tensor(self.output_dim, self.output_dim))

        if bias:
            self.bias = Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.output_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, adjacency_matrix, root_embeddings):
        if self.transfer:
            node_features = node_features.matmul(self.weight)
        else:
            node_features = node_features.matmul(self.weight2)

        if self.bias is not None:
            node_features = node_features + self.bias

        adjacency_matrix_t = F.softmax(adjacency_matrix.T, dim=1)
        edge_embeddings = torch.matmul(adjacency_matrix_t, node_features)

        edge_embeddings = F.dropout(edge_embeddings, self.dropout, training=self.training)
        edge_embeddings = F.relu(edge_embeddings, inplace=False)

        edge_transformed = edge_embeddings.matmul(self.weight3)

        adjacency_matrix = F.softmax(adjacency_matrix, dim=1)

        node_attention_embeddings = torch.matmul(adjacency_matrix, edge_transformed)
        node_attention_embeddings = F.dropout(node_attention_embeddings, self.dropout, training=self.training)

        if self.concat:
            node_attention_embeddings = F.relu(node_attention_embeddings, inplace=False)

        if self.use_edge:
            edge_output = torch.matmul(adjacency_matrix_t, node_attention_embeddings)
            edge_output = F.dropout(edge_output, self.dropout, training=self.training)
            edge_output = F.relu(edge_output, inplace=False)
            return node_attention_embeddings, edge_output
        else:
            return node_attention_embeddings

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.input_dim} -> {self.output_dim})"