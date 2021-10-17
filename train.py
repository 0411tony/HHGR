import argparse
import time
import gc
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.HHGR import HHGR
from utils.user_utils import TrainUserDataset, EvalUserDataset
from utils.group_utils import TrainGroupDataset, EvalGroupDataset
from utils.dataset import GDataset
from utils import util
from utils.util import Helper

if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_id = int(np.argmax(memory_available))
    torch.cuda.set_device(gpu_id)

parser = argparse.ArgumentParser(description='PyTorch HHGR: Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation')
parser.add_argument('--dataset', type=str, default='weeplaces', help='Name of dataset')

# Training settings.
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--lambda_mi', type=float, default=1.0, help='MI lambda hyper param')
parser.add_argument('--drop_ratio', type=float, default=0.4, help='Dropout ratio')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=8, help='maximum # training epochs')

# Model settings.
parser.add_argument('--emb_size', type=int, default=64, help='layer size')
parser.add_argument('--negs_per_group', type=int, default=10, help='# negative users sampled per group')
# parser.add_argument('--pretrain_epochs', type=int, default=100, help='# pre-train epochs for user encoder layer')

parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')

args = parser.parse_args()

torch.manual_seed(args.seed)  # Set the random seed manually for reproducibility.

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###############################################################################
# Load data
###############################################################################
train_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 5, 'pin_memory': True}
eval_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 5, 'pin_memory': True}
device = torch.device("cuda" if args.cuda else "cpu")

# Define train/val/test datasets on user interactions.
train_dataset = TrainUserDataset(args.dataset)  # train dataset for user-item interactions.
n_items, n_users = train_dataset.n_items, train_dataset.n_users
data_gu_dict = train_dataset.data_gu_dict
val_dataset = EvalUserDataset(args.dataset, n_items, datatype='val')
test_dataset = EvalUserDataset(args.dataset, n_items, datatype='test')

# Define train/val/test datasets on group and user interactions.
train_group_dataset = TrainGroupDataset(args.dataset, n_items, args.negs_per_group)
padding_idx = train_group_dataset.padding_idx
val_group_dataset = EvalGroupDataset(args.dataset, n_items, padding_idx, datatype='val')
test_group_dataset = EvalGroupDataset(args.dataset, n_items, padding_idx, datatype='test')
n_groups = train_group_dataset.n_group_gu
ui_matrix, gi_matrix = train_dataset.train_data_ui.todok(), train_group_dataset.group_data.todok()
gi_matrix_dense = train_group_dataset.group_data.todense()
dataset = GDataset(gi_matrix_dense, ui_matrix, gi_matrix, args.negs_per_group)

# Define data loaders on user interactions.
train_loader = DataLoader(train_dataset, **train_params)
fine_data_ug, coarse_data_ug = train_dataset.fine_data_ug.toarray(), train_dataset.coarse_data_ug.toarray()
###########################################################################################
data_gg_csr = train_group_dataset.H_gg.toarray()
num_users_gu = train_dataset.n_users_gu

H_ul_coarse = util.generate_G_from_H(coarse_data_ug)
H_ul_coarse = torch.Tensor(H_ul_coarse)

H_ul_fine = util.generate_G_from_H(fine_data_ug)
H_ul_fine = torch.Tensor(H_ul_fine)

H_gl = torch.Tensor(data_gg_csr)

all_uid = [i for i in range(n_users)]
all_group_inputs = []
for i in range(n_groups):
    all_group_inputs.append(i)

###############################################################################
# Build the model
###############################################################################
user_layers = [args.emb_size]  # user encoder layer configuration is tunable.
helper = Helper()
model = HHGR(n_items, n_users, n_groups, data_gu_dict, user_layers, drop_ratio=args.drop_ratio, lambda_mi=args.lambda_mi).to(device)

torch.autograd.set_detect_anomaly(True)

print("args", args)
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer_ur = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=args.wd)
    pre_train_loader = dataset.get_user_dataloader(args.batch_size)
    print("Pre-training model on user-item interactions")
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        model.train()

        start_time = time.time()
        losses = []
        for batch_index, (u, pi_ni) in enumerate(pre_train_loader):
            user_input = u
            pos_item_input = pi_ni[:, 0]
            neg_item_input = pi_ni[:, 1]
            pos_prediction = model.usr_forward(user_input, pos_item_input)
            neg_prediction = model.usr_forward(user_input, neg_item_input)

            model.zero_grad()
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
            losses.append(float(loss))
            loss.backward(retain_graph=True)
            optimizer_ur.step()
            del pos_item_input, neg_item_input, pos_prediction, neg_prediction, user_input
        gc.collect()
        elapsed = time.time() - start_time
        print('| epoch {:3d} |  time {:4.2f} | loss {:4.2f}'.format(epoch + 1, elapsed, np.mean(losses)))

    print("loading the user-level hypergraph learning")
    optimizer_ul = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=args.wd)
    all_user_embedding = model.userembedds(torch.LongTensor(all_uid)).detach()
    f_all_user_embedding = all_user_embedding
    for epoch in range(0, args.epochs):
        model.train()
        start_time = time.time()
        user_level_loss = []
        user_embed_coarse = model.hgcn_coarse(all_user_embedding, H_ul_coarse)
        user_embed_fine = model.hgcn_fine(f_all_user_embedding, H_ul_fine)
        for batch_index, data in enumerate(train_loader):
            (uid, neg_id, train_users, train_items) = data  # train_items (256, 25081)
            optimizer_ul.zero_grad()
            model.zero_grad()
            model.train()
            uembed_coarse = user_embed_coarse[uid]
            uembed_fine = user_embed_fine[uid]
            neg_uembed = user_embed_coarse[neg_id]

            scores_pos = model.discriminator(uembed_coarse, uembed_fine)  # positive sample
            scores_neg = model.discriminator(uembed_coarse, neg_uembed)  # negative sample
            mi_loss = model.discriminator.mi_loss(scores_pos, scores_neg)
            with torch.autograd.detect_anomaly():
                mi_loss.backward(retain_graph=True)
            user_level_loss.append(float(mi_loss))

            optimizer_ul.step()
            del uid, neg_id, train_users, train_items, uembed_coarse, uembed_fine, neg_uembed
        gc.collect()
        elapsed = time.time() - start_time
        print('| epoch {:3d} |  time {:4.2f} | loss {:4.2f}'.format(epoch + 1, elapsed, np.mean(user_level_loss)))
    
    ##### group-level hypergraph #######
    print("loading the group-level hypergraph learning")
    all_user_embedding = model.userembedds(torch.LongTensor(all_uid)).detach()
    f_all_user_embedding = model.userembedds(torch.LongTensor(all_uid)).detach()
    user_embed_coarse = model.hgcn_coarse(all_user_embedding, H_ul_coarse).detach()
    user_embed_fine = model.hgcn_fine(f_all_user_embedding, H_ul_fine).detach()
    user_embedding = user_embed_coarse + user_embed_fine

    del user_embed_coarse, user_embed_fine

    optimizer_gr = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.wd)
    group_loader = dataset.get_group_dataloader(512)
    print(len(group_loader))
    for epoch in range(0, 1):
        model.train()
        start_time = time.time()
        group_level_loss = []
        gr_embedds = torch.Tensor()
        all_group_embdding = model.groupembeds(torch.LongTensor(all_group_inputs))
        group_embedds = model.hgcn_gl(all_group_embdding, H_gl)
        for batch_id, (g, pi_ni) in enumerate(group_loader):
            if batch_id % 10 == 0:
                print('group hypergraph | batch id: {:3d}'.format(batch_id+1))
            group_input = g
            pos_item_input = pi_ni[:, 0]
            neg_item_input = pi_ni[:, 1]
            optimizer_gr.zero_grad()
            model.zero_grad()
            model.train()

            group_embedd, pos_prediction = model.train_grp_forward(group_input, pos_item_input, user_embedding, group_embedds)
            neg_group_embedd, neg_prediction = model.train_grp_forward(group_input, neg_item_input, user_embedding, group_embedds)

            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
            group_level_loss.append(float(loss))
            with torch.autograd.detect_anomaly():
                loss.backward(retain_graph=True)
            optimizer_gr.step()

            del pos_prediction, neg_prediction, pos_item_input, neg_item_input

        gc.collect()
        elapsed = time.time() - start_time
        print('| epoch {:3d} |  time {:4.2f} | loss {:4.2f}'.format(epoch + 1, elapsed, np.mean(group_level_loss)))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

######################## evaluate the model ##############################
all_user_embedding = model.userembedds(torch.LongTensor(all_uid)).detach()
fine_user_embedding = all_user_embedding
user_embedds_coarse = model.hgcn_coarse(all_user_embedding, H_ul_coarse)
user_embedds_fine = model.hgcn_fine(fine_user_embedding, H_ul_fine)
user_embedds_ul = user_embedds_coarse + user_embedds_fine
hits, ndcgs = helper.evaluate_model(model, test_dataset.ui_list, test_dataset.ui_neg_list, user_embedds_ul, None, 20, 'user')
u_hr, u_ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
hits_50, ndcgs_50 = helper.evaluate_model(model, test_dataset.ui_list, test_dataset.ui_neg_list, user_embedds_ul, None, 50, 'user')
u_hr_50, u_ndcg_50 = np.array(hits_50).mean(), np.array(ndcgs_50).mean()
print('user test-top 20 | HR = %.4f, NDCG = %.4f' % (u_hr, u_ndcg))
print('user test-top 50 | HR = %.4f, NDCG = %.4f' % (u_hr_50, u_ndcg_50))
# user_embedds_ul = model.userembedds(torch.LongTensor(all_uid)).detach()

data_gl_test = test_group_dataset.H_gg_test.toarray()
H_gl_test = torch.Tensor(data_gl_test)
all_group_inputs = [i for i in range(test_group_dataset.n_groups_test)]
all_group_embedds = model.groupembeds(torch.LongTensor(all_group_inputs)).detach()
test_group_embedds = model.hgcn_gl(all_group_embedds, H_gl_test)
g_hits, g_ndcgs = helper.evaluate_model(model, test_group_dataset.gid, test_group_dataset.nid, user_embedds_ul, test_group_embedds, 20, 'group')
g_hr, g_ndcg = np.array(g_hits).mean(), np.array(g_ndcgs).mean()
print('group test-top 20 | HR = %.4f, NDCG = %.4f' % (g_hr, g_ndcg))
g_hits_50, g_ndcgs_50 = helper.evaluate_model(model, test_group_dataset.gid, test_group_dataset.nid, user_embedds_ul, test_group_embedds, 50, 'group')
g_hr_50, g_ndcg_50 = np.array(g_hits_50).mean(), np.array(g_ndcgs_50).mean()
print('group test-top 50 | HR = %.4f, NDCG = %.4f' % (g_hr_50, g_ndcg_50))
