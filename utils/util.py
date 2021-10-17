'''
Created on March 10, 2021
Deal something

@author: Junwei Zhang
'''
import torch
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import math
import heapq

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    DV = np.sum(H, axis=1) + 1e-5
    DE = np.sum(H, axis=0) + 1e-5

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -1)))
    H = np.mat(H)
    HT = H.T

    G = DV2 * H * invDE * HT * DV2
    return G

def matrix_to_sp_tensor(H):
    H_coo = H.tocoo()

    indices = []
    indices.append(list(H_coo.row))
    indices.append(list(H_coo.col))
    indices = torch.LongTensor(indices)
    values = torch.LongTensor(list(H_coo.data))

    tensor_sprse = torch.sparse_coo_tensor(indices=indices, values=values, size=H_coo.shape)
    return tensor_sprse

def G_from_H_sp(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence sparse matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    DV = np.sum(H, axis=1) + 1e-5
    DE = np.sum(H, axis=0) + 1e-5

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -1)))
    H = np.mat(H)
    HT = H.T

    G = DV2 * H * invDE * HT * DV2
    G_array = np.array(G)
    row, col = np.nonzero(G_array)
    row, col = list(row), list(col)
    data = []
    for i in range(len(row)):
        a = row[i]
        b = col[i]
        data.append(G_array[a][b])
    adj = sp.coo_matrix((data, (row, col)), shape=(len(G_array), len(G_array)))

    H_coo = adj
    indices = []
    indices.append(row)
    indices.append(col)
    indices = torch.LongTensor(indices)
    values = torch.LongTensor(data)

    tensor_sprse = torch.sparse_coo_tensor(indices=indices, values=values, size=H_coo.shape)
    return tensor_sprse


class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def test_evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        for idx in range(len(testRatings)):
            (hr, ndcg) = self.test_eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        group_hit_counts = len(np.nonzero(hits)[0])
        return (hits, ndcgs)

    def test_eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users)
        users_var = users_var.long()
        items_var = torch.LongTensor(items)
        if type_m == 'group':
            predictions = model.train_grp_forward(users_var, items_var)
        elif type_m == 'user':
            predictions = model.usr_forward(users_var, items_var)

        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def evaluate_model(self, model, testRatings, testNegatives, all_embeds, group_embedds, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            if idx%10 == 0:
                print('test idx | batch id: {:3d} / {:3d}'.format(idx+1, len(testRatings)))
            (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, all_embeds, group_embedds, K, type_m, idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        group_hit_counts = len(np.nonzero(hits)[0])
        return (hits, ndcgs)

    def eval_one_rating(self, model, testRatings, testNegatives, all_embeds, group_embedds, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users)
        users_var = users_var.long()
        items_var = torch.LongTensor(items)
        if type_m == 'group':
            new_embeds, predictions = model.train_grp_forward(users_var, items_var, all_embeds, group_embedds)
        elif type_m == 'user':
            predictions = model.user_forward(users_var, items_var, all_embeds)
            # predictions = model.usr_forward(users_var, items_var)

        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)

        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0