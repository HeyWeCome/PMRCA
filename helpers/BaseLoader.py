import logging
import random
import numpy as np
import torch
from torch.autograd import Variable
from utils import Constants
import pickle

class BaseLoader(object):
    def __init__(self, args):
        self.data_name = args.data_name
        self.data = 'data/' + self.data_name + '/cascades.txt'
        self.u2idx_dict = 'data/' + self.data_name + '/u2idx.pickle'
        self.idx2u_dict = 'data/' + self.data_name + '/idx2u.pickle'
        self.save_path = ''
        self.net_data = 'data/' + self.data_name + '/edges.txt'
        self.user_num = 0
        self.cas_num = 0
        self.min_cascade_len = args.filter_num

    def split_data(self, train_ratio=0.8, valid_ratio=0.1, load_dict=True, with_eos=True):
        user_to_index = {}
        index_to_user = []

        if not load_dict:
            user_count, user_to_index, index_to_user = self.build_index(self.data)
            with open(self.u2idx_dict, 'wb') as f:
                pickle.dump(user_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.idx2u_dict, 'wb') as f:
                pickle.dump(index_to_user, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.u2idx_dict, 'rb') as f:
                user_to_index = pickle.load(f)
            with open(self.idx2u_dict, 'rb') as f:
                index_to_user = pickle.load(f)
            user_count = len(user_to_index)

        cascades, timestamps = [], []

        for line in open(self.data):
            if not line.strip():
                continue

            current_timestamps = []
            current_users = []
            chunks = line.strip().split(',')

            for chunk in chunks:
                try:
                    parts = chunk.split()
                    if len(parts) == 2:
                        user, timestamp = parts
                    elif len(parts) == 3:
                        root, user, timestamp = parts
                        if root in user_to_index:
                            current_users.append(user_to_index[root])
                            current_timestamps.append(float(timestamp))

                except Exception as e:
                    print(f"Error processing chunk: {chunk} -> {e}")

                if user in user_to_index:
                    current_users.append(user_to_index[user])
                    current_timestamps.append(float(timestamp))

            if self.min_cascade_len <= len(current_users) <= 500:
                if with_eos:
                    current_users.append(Constants.EOS)
                    current_timestamps.append(Constants.EOS)
                cascades.append(current_users)
                timestamps.append(current_timestamps)

        sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
        cascades = [cascades[i] for i in sorted_indices]
        timestamps = sorted(timestamps)

        total_cascades = len(cascades)
        train_end = int(train_ratio * total_cascades)
        valid_end = int((train_ratio + valid_ratio) * total_cascades)

        train_set = cascades[:train_end], timestamps[:train_end], sorted_indices[:train_end]
        valid_set = (
            cascades[train_end:valid_end],
            timestamps[train_end:valid_end],
            sorted_indices[train_end:valid_end]
        )
        test_set = cascades[valid_end:], timestamps[valid_end:], sorted_indices[valid_end:]

        for data in train_set:
            random.shuffle(data)

        total_length = sum(len(cas) for cas in cascades)
        logging.info(
            f"Training size: {len(train_set[1])}\nValidation size: {len(valid_set[1])}\nTesting size: {len(test_set[1])}")
        logging.info(f"Total size: {total_cascades}\nAverage length: {total_length / total_cascades:.2f}")
        logging.info(f"Maximum length: {max(len(cas) for cas in cascades):.2f}")
        logging.info(f"Minimum length: {min(len(cas) for cas in cascades):.2f}")

        self.user_num = user_count
        self.cas_num = len(cascades)
        logging.info(f"User size: {self.user_num - 2}")
        logging.info(f"Cascade size: {self.cas_num}")

        self.cascades = cascades
        self.timestamps = timestamps
        self.train_data = train_set
        self.valid_data = valid_set
        self.test_data = test_set
        self.train_cas_user_dict = self.create_cascade_user_dict(train_set)
        return user_count, cascades, timestamps, train_set, valid_set, test_set

    def create_cascade_user_dict(self, train_set):
        cascades, _, indices = train_set
        cascade_dict = {}

        for i, cascade in zip(indices, cascades):
            user_list = [user for user in cascade if user != Constants.EOS]
            cascade_dict[i] = user_list

        return cascade_dict

    def build_index(self, data):
        user_set = set()
        u2idx = {}
        idx2u = []

        line_id = 0
        for line in open(data):
            line_id += 1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    if len(chunk.split()) == 2:
                        user, timestamp = chunk.split()
                    elif len(chunk.split()) == 3:
                        root, user, timestamp = chunk.split()
                        user_set.add(root)
                except:
                    logging.error(line)
                    logging.error(chunk)
                    logging.error(line_id)
                user_set.add(user)

        pos = 0
        u2idx['<blank>'] = pos
        idx2u.append('<blank>')
        pos += 1
        u2idx['</s>'] = pos
        idx2u.append('</s>')
        pos += 1

        for user in user_set:
            u2idx[user] = pos
            idx2u.append(user)
            pos += 1

        user_size = len(user_set) + 2
        logging.info("user_size : %d" % (user_size - 2))
        return user_size, u2idx, idx2u

class DataLoader(object):
    """ 用于数据迭代的类 """

    def __init__(
            self, cas, batch_size=64, load_dict=True, cuda=True, test=False, with_EOS=True):
        self._batch_size = batch_size
        self.cas = cas[0]
        self.time = cas[1]
        self.idx = cas[2]
        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda

        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        """ 获取下一个批处理 """

        def pad_to_longest(insts):
            """ 将实例填充到批处理中最长序列长度 """

            max_len = 500

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            seq_idx = Variable(
                torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)

            return seq_data, seq_data_timestamp, seq_idx
        else:
            self._iter_count = 0
            raise StopIteration()