import gc
import os

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils import data
import random

class TrainUserDataset(data.Dataset):
    """ Train User Data Loader: load training user-item interactions """

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = os.path.join('data/', dataset)
        self.train_data_ui = self._load_train_data()
        self.fine_data_ug, self.coarse_data_ug, self.data_gu_dict = self._load_group_data()
        self.user_list = list(range(self.n_users))

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        """ load user_id, binary vector over items """
        user = self.user_list[index]
        user_items = torch.from_numpy(self.train_data_ui[user, :].toarray()).squeeze()  # [I]
        neg = random.randint(0, len(self.user_list)-1)
        while neg == user:
            neg = random.randint(0, len(self.user_list) - 1)
        neg_user = self.user_list[neg]

        return user, neg_user, torch.from_numpy(np.array([user], dtype=np.int32)), user_items

    def _load_train_data(self):
        """ load training user-item interactions as a sparse matrix """
        path_ui = os.path.join(self.data_dir, 'train_ui.csv')
        df_ui = pd.read_csv(path_ui)
        self.n_users, self.n_items = df_ui['user'].max() + 1, df_ui['item'].max() + 1
        self.start_idx, self.end_idx = df_ui['user'].min(), df_ui['user'].max()
        rows_ui, cols_ui = df_ui['user'], df_ui['item']
        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.n_users, self.n_items))  # [# train users, I] sparse matrix

        print("# train users", self.n_users, "# items", self.n_items)

        return data_ui

    def _load_group_data(self):
        """ load training group-item interactions as a sparse matrix and user-group memberships """
        print("load train group-user dataset")
        path_ug = os.path.join(self.data_dir, 'group_users.csv')
        df_ug = pd.read_csv(path_ug).astype(int)  # load user-group memberships.
        df_ug_train = df_ug[df_ug.user.isin(range(self.start_idx, self.end_idx + 1))]

        self.n_users_gu = df_ug_train['user'].max() + 1  # user number: 8643
        self.n_group_gu = df_ug_train['group'].max() - df_ug_train['group'].min() + 1  # group number: 22733
        rows_gu, cols_gu = df_ug_train['user'], df_ug_train['group'] - df_ug_train['group'].min()
        array = np.ones_like(rows_gu)

        # group-user relationship incidence matrix / size: (num_group, num_user) (22733, 8643)
        data_gu_csr = sp.csr_matrix((array, (rows_gu, cols_gu)), dtype='float32',
                                    shape=(self.n_users_gu, self.n_group_gu))

        #################### generate the dict of user and group  ##########################
        data_gu_denser = data_gu_csr.toarray().transpose()
        data_gu_dict = {}
        for g in range(len(data_gu_denser)):
            data_gu_dict[g] = []
            for u in range(len(data_gu_denser[g])):
                data_gu_dict[g].append(int(u))

        #################### generate the coarse and fine hypergraph ##########################
        Theta = np.zeros(self.n_users_gu, dtype=int)
        rand = np.random.randint(0, len(Theta), int(0.2 * len(Theta)))
        if (len(set(rand)) != len(rand)):
            rand = np.random.randint(0, len(Theta), int(0.2 * len(Theta)))
        # print(rand)
        for i in rand:
            Theta[i] = 1
        data_gu_csr_coarse = data_gu_csr.toarray()
        data_gu_csr_coarse = data_gu_csr_coarse.transpose() * (np.array(Theta.transpose()))
        data_gu_csr_coarse = data_gu_csr_coarse.transpose()
        data_gu_csr_coarse = sp.csr_matrix(data_gu_csr_coarse)

        Beta = np.zeros(self.n_group_gu, dtype=int)
        data_gu_csr_fine = data_gu_csr.toarray()
        rand = np.random.randint(0, len(Beta), int(0.3 * len(Beta)))

        for i in range(len(data_gu_csr_fine)):
            if (len(set(rand)) != len(rand)):
                rand = np.random.randint(0, len(Beta), int(0.3 * len(Beta)))
            for j in rand:
                Beta[j] = 1
            data_gu_csr_fine[i] = data_gu_csr_fine[i] * (np.array(Beta.transpose()))
        data_gu_csr_fine = sp.csr_matrix(data_gu_csr_fine)

        return data_gu_csr_fine, data_gu_csr_coarse, data_gu_dict


class EvalUserDataset(data.Dataset):
    """ Eval User Data Loader: load val/test user-item interactions with fold-in and held-out item sets """

    def __init__(self, dataset, n_items, datatype='val'):
        self.dataset = dataset
        self.n_items = n_items
        self.data_dir = os.path.join('data/', dataset)
        self.data_tr, self.data_te, self.ui_list, self.ui_neg_list, self.test_uid = self._load_tr_te_data(datatype)

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        """ load user_id, fold-in items, held-out items """
        user = self.user_list[index]
        fold_in, held_out = self.data_tr[user, :].toarray(), self.data_te[user, :].toarray()  # [I], [I]
        return user, torch.from_numpy(fold_in).squeeze(), held_out.squeeze()  # user, fold-in items, fold-out items.

    def _load_tr_te_data(self, datatype='val'):
        """ load user-item interactions of val/test user sets as two sparse matrices of fold-in and held-out items """
        # self.data_dir = os.path.join('data/', dataset)
        ui_tr_path = os.path.join(self.data_dir, '{}_ui_tr.csv'.format(datatype))
        ui_te_path = os.path.join(self.data_dir, '{}_ui_te.csv'.format(datatype))

        ui_df_tr, ui_df_te = pd.read_csv(ui_tr_path), pd.read_csv(ui_te_path)

        start_idx = min(ui_df_tr['user'].min(), ui_df_te['user'].min())
        end_idx = max(ui_df_tr['user'].max(), ui_df_te['user'].max())

        rows_tr, cols_tr = ui_df_tr['user'] - start_idx, ui_df_tr['item']
        rows_te, cols_te = ui_df_te['user'] - start_idx, ui_df_te['item']
        self.user_list = list(range(0, end_idx - start_idx + 1))

        ui_data_tr = sp.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.n_items))  # [# eval users, I] sparse matrix
        ui_data_te = sp.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.n_items))  # [# eval users, I] sparse matrix

        ui_te_dense = ui_data_te.todense()

        ui_list = []
        ui_neg_list = []
        for i in range(len(ui_te_dense)):
            uid = int(i)
            for j in range(len(ui_te_dense[i])):
                iid = int(j)
                ui_list.append([uid, iid])
                neg_list = list(np.random.randint(0, self.n_items, 100))
                ui_neg_list.append(neg_list)
            del uid, iid, neg_list
        gc.collect()

        return ui_data_tr, ui_data_te, ui_list, ui_neg_list, end_idx
