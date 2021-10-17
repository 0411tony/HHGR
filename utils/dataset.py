import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class GDataset(object):

    def __init__(self, gi_matrix_dense, ui_matrix, gi_matrix, num_negatives):
        self.num_negatives = num_negatives
        self.ui_matrix = ui_matrix
        self.group_trainMatrix = gi_matrix
        self.gr_matrix = gi_matrix_dense

    def get_group(self, train):
        group_input, pos_item_input = [], []
        num_groups = train.shape[0]
        num_items = train.shape[1]
        for i in range(num_groups):
            pos_item_input.append(list(np.nonzero(train[i])[0]))
            group_input.append([group_input]*len(np.nonzero(train[i])[0]))
            group_input.append(i)
        return group_input, pos_item_input

    def get_train_group(self, train):
        group_input, pos_item_input, neg_item_input = [], [], []
        num_items = train.shape[1]
        for (g, i) in train.keys():
            group_input.append(g)
            pos_item_input.append(i)
            j = np.random.randint(num_items)
            while (g, j) in train:
                j = np.random.randint(num_items)
            neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return group_input, pi_ni

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.ui_matrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_group(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader

    def get_gr_dataloader(self, batch_size):
        group, positem_at_g = self.get_group(self.gr_matrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader
