import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator for Mutual Information Estimation and Maximization, implemented with bilinear layers and
    binary cross-entropy loss training """

    def __init__(self, embedding_dim=64):
        super(Discriminator, self).__init__()
        # self.n_users = n_users
        self.embedding_dim = embedding_dim

        self.fc_layer = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.embedding_dim, self.embedding_dim, 1)  # output_dim = 1 => single score.
        nn.init.zeros_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pos_inputs, neg_inputs):
        """ bilinear discriminator:
            :param pos_inputs: [B, I]
            :param neg_inputs: [B, I]
        """
        # FC + activation.
        coarse_encoded = self.fc_layer(pos_inputs)  # [B, D]
        # FC + activation.
        fine_encoded = self.fc_layer(neg_inputs)  # [B, D]

        return self.bilinear_layer(coarse_encoded, fine_encoded)

    def mi_loss(self, scores_group, scores_corrupted, device='cpu'):
        """ binary cross-entropy loss over (group, user) pairs for discriminator training
            :param scores_group: [B, G]
            :param scores_corrupted: [B, N]
            :param device (cpu/gpu)
         """
        batch_size = scores_group.shape[0]
        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)  # [B, G]
        zero_labels = torch.zeros(batch_size, neg_size).to(device)  # [B, N]

        labels = torch.cat((one_labels, zero_labels), 1)  # [B, G+N]
        logits = torch.cat((scores_group, scores_corrupted), 1)  # [B, G + N]

        mi_loss = self.bce_loss(logits, labels) * (batch_size * (pos_size + neg_size)) \
                  / (batch_size * neg_size)

        return mi_loss
