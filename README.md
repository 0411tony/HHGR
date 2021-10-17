# RHINE
Source code for CIKM 2021 paper ["**Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation**"](https://arxiv.org/abs/2109.04200)

# Requirements

- Python 3.8
- PyTorch (1.9.1)
- numpy (1.19.2)
- pandas (1.2.4)
- scipy (1.6.2)
- sklearn (0.24.2)


# Description

```
HHGR-s2/
├── models
│   ├── HHGR.py: the main model with some functions and configs for the model
│   ├── HGCN.py: the hypergraph convolutional network model
│   ├── Discriminator.py: discriminator network model for self-supervised learning
│   ├── EmbeddingLayer.py: Embedding network model for learning the representations of group, user, and item
├── utils
│   ├── util.py: evaluate the performance of learned embeddings w.r.t clustering and classification
│   ├── dataset.py: generate the group and user dataloader 
│   ├── user_tuils.py: generate the user dataloader for training the model
│   ├── group_tuils.py: generate the group dataloader for training the model
├── data
│   └── weeplaces
│       ├── group_users.csv: the group-user relationship
│       ├── train_ui.csv: the training file of user-item history interaction
│       ├── train_gi.csv: the training file of group-item history interaction
│       ├── val_ui.csv: the validation file of user-item history interaction
│       ├── val_gi.csv: the validation file of group-item history interaction
│       ├── test_ui.csv: the test file of user-item history interaction
│       ├── test_gi.csv: the test file of user-item history interaction
├── README.md
```

# Reference

```
@article{DBLP:journals/corr/abs-2109-04200,
  author    = {Junwei Zhang, Min Gao, Junliang Yu, Lei Guo, Jundong Li, and Hongzhi Yin},
  title     = {Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation},
  booktitle={Proceedings of CIKM},
  year      = {2021},
}

```
