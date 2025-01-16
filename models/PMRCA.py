import math
import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.data import Data
from layers import GraphBuilder
from utils import Constants
from torch.autograd import Variable
from layers.Commons import GraphNN, Fusion
from layers.TransformerBlock import TransformerBlock

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class PMRCA(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--pos_dim', type=int, default=64)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--time_step_split', type=int, default=8)
        parser.add_argument('--gcn_layers', type=int, default=3)
        parser.add_argument('--ssl_reg', type=float, default=1e-7)
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--ssl_temp', type=float, default=0.1)
        parser.add_argument('--K', type=int, default=2)
        parser.add_argument('--nc', type=int, default=1)
        return parser

    def __init__(self, args, data_loader):
        super(PMRCA, self).__init__()
        self.args = args  # Store args
        self.device = 'cuda'
        self.user_num = data_loader.user_num
        self.cas_num = data_loader.cas_num
        self.embedding_size = args.d_model
        self.pos_dim = args.pos_dim
        self.n_heads = args.n_heads
        self.drop_timestamp = nn.Dropout(args.dropout)
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.cas_embedding = nn.Embedding(self.cas_num, self.embedding_size)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)
        self.align_attention = TransformerBlock(input_size=self.embedding_size, n_heads=self.n_heads)
        self.linear = nn.Linear(self.embedding_size, self.user_num)

        train_cas_user_dict = data_loader.train_cas_user_dict
        self.norm_adj = self.csr2tensor(self.build_adjmat(self.cas_num,
                                                          self.user_num,
                                                          train_cas_user_dict))
        self.gcn_layers = args.gcn_layers
        self.ssl_reg = args.ssl_reg
        self.alpha = args.alpha
        self.ssl_temp = args.ssl_temp
        self.proto_reg = 8e-8
        self.num_clusters = 10
        self.attn_size = 8
        self.K = args.K
        self.W1 = nn.Linear(self.embedding_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)
        self.linear2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.k = self.num_clusters
        self.user_centroids = None
        self.user_2cluster = None
        self.cas_centroids = None
        self.cas_2cluster = None
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        cas_embeddings = self.cas_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.cas_centroids, self.cas_2cluster = self.run_kmeans(cas_embeddings)

    def run_kmeans(self, x):
        import faiss
        kmeans = faiss.Kmeans(d=self.embedding_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        if selfloop_flag:
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        return adj_mat

    def csr2tensor(self, matrix):
        rowsum = np.array(matrix.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(matrix).dot(d_mat_inv_sqrt)

        coo_bi_lap = bi_lap.tocoo()

        row = coo_bi_lap.row
        col = coo_bi_lap.col 
        data = coo_bi_lap.data

        i = torch.LongTensor(np.array([row, col]))
        data_tensor = torch.from_numpy(data).float()

        sparse_bi_lap = torch.sparse_coo_tensor(i, data_tensor, torch.Size(coo_bi_lap.shape), dtype=torch.float32)

        if self.device == 'cuda':
            norm_adj_mat = sparse_bi_lap.to(self.device)
            return norm_adj_mat
        else:
            norm_adj_mat = sparse_bi_lap.to_dense().to(self.device)
            return norm_adj_mat

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        cas_embeddings = self.cas_embedding.weight
        ego_embeddings = torch.cat([cas_embeddings, user_embeddings], dim=0)
        return ego_embeddings

    def ssl_layer_loss(self, current_embedding, previous_embedding, cas_idx, user_idx):
        current_cas_embeddings, current_user_embeddings = torch.split(
            current_embedding, [self.cas_num, self.user_num]
        )
        previous_cas_embeddings_all, previous_user_embeddings_all = torch.split(
            previous_embedding, [self.cas_num, self.user_num]
        )

        current_user_embeddings = current_user_embeddings[user_idx]
        previous_user_embeddings = previous_user_embeddings_all[user_idx]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_cas_embeddings[cas_idx]
        previous_item_embeddings = previous_cas_embeddings_all[cas_idx]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_cas_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def aware_loss(self, node_embedding, cas, tg_user):
        cas_embeddings_all, tuser_embeddings_all = torch.split(
            node_embedding, [self.cas_num, self.user_num]
        )

        user_embeddings = tuser_embeddings_all[tg_user]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[tg_user]
        user2centroids = self.user_centroids[user2cluster]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(
            norm_user_embeddings, self.user_centroids.transpose(0, 1)
        )
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = cas_embeddings_all[cas]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.cas_2cluster[cas]
        item2centroids = self.cas_centroids[item2cluster]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(
            norm_item_embeddings, self.cas_centroids.transpose(0, 1)
        )
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def forward(self, input_seq, input_timestamp, tgt_idx):
        ego_embeddings = self.get_ego_embeddings().to(self.device)
        embedding_list = [ego_embeddings]

        for layer_idx in range(self.gcn_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            embedding_list.append(ego_embeddings)

        gcn_all_embeddings = torch.stack(embedding_list, dim=1)
        gcn_all_embeddings = torch.mean(gcn_all_embeddings, dim=1)

        cas_all_embeddings, user_all_embeddings = torch.split(
            gcn_all_embeddings, [self.cas_num, self.user_num]
        )

        input_seq = input_seq[:, :-1]
        mask = (input_seq == Constants.PAD)

        batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).cuda()
        position_embed = self.pos_embedding(batch_t)

        batch_size, max_len = input_seq.size()

        original_seq_emb = self.user_embedding(input_seq.cuda())
        dyemb = user_all_embeddings[input_seq.cuda()]
        dyemb += position_embed

        valid_his = (input_seq > 0).long().cuda()


        attn_score = self.W2(self.W1(original_seq_emb).tanh()).cuda()
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf).cuda()
        attn_score = attn_score.transpose(-1, -2).cuda()
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1).cuda()
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0).cuda()
        intention_vectors = (original_seq_emb[:, None, :, :] * attn_score[:, :, :, None]).sum(-2).sum(
            -2).cuda()

        intention_vectors = intention_vectors.unsqueeze(1)
        intention_vectors = intention_vectors.expand(-1, original_seq_emb.size(1), -1)

        att_out = self.align_attention(dyemb.cuda(),
                                        intention_vectors.cuda(),
                                        intention_vectors.cuda(),
                                        mask=mask.cuda())

        output = self.linear(att_out.cuda())

        mask = self.get_previous_user_mask(input_seq.cuda(), self.user_num)
        output = output.cuda() + mask.cuda()

        if self.training:
            return output.view(-1, output.size(-1)), user_all_embeddings, cas_all_embeddings, embedding_list
        else:
            return output.view(-1, output.size(-1))

    def get_previous_user_mask(self, seq, user_size):
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()
        masked_seq = previous_mask * seqs.data.float()

        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq.cuda()

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        pred, user_all_embeddings, cas_all_embeddings, embedding_list = self.forward(input_seq,
                                                                                     input_seq_timestamp,
                                                                                     history_seq_idx)

        center_embedding = embedding_list[0]
        context_embedding = embedding_list[1]

        ssl_loss = torch.tensor(0.0).cuda()
        proto_loss = torch.tensor(0.0).cuda()

        ssl_loss = self.ssl_layer_loss(
            context_embedding, center_embedding, history_seq_idx, gold[:, -1]
        )
    
        proto_loss = self.aware_loss(center_embedding, history_seq_idx, gold[:, -1])

        loss = loss_func(pred, gold.contiguous().view(-1))

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
        return loss + ssl_loss + proto_loss, n_correct

    def before_epoch(self):
        self.e_step()