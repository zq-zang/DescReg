from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        ##todo
        # mask = torch.eq(labels, labels.T).float().to(device)
        mask = torch.eq(labels, labels.transpose(1, 0)).float().to(device)
        #

        ##todo
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(features, features.T),
        #     self.temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.transpose(1, 0)),
            self.temperature)
        ##
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, adj_matrix_file, pos_thr=0.2, loss_scale=0.5, margin_scale=2.0, unseen_class_ids=None, temperature=0.07, eps=1e-7):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature
        self.eps = 1e-7
        # self.group_labels = torch.tensor([0, 4, 5, 1, 5, 1, 1, 5, 6, 1, 1, 4, 2, 5, 3, 1, 4, 1, 3, 2, 4]).cuda()# dior/visdrone
        # self.group_labels = torch.tensor([0, 2, 8, 9, 6, 1, 4, 7, 7, 3, 3, 6, 1, 2, 4, 5]).cuda()# dota
        if adj_matrix_file is not None:
            self.adj_matrix = torch.tensor(np.load(adj_matrix_file)).float().cuda()
            if unseen_class_ids is not None:
                unseen_class_ids = [u + 1 for u in unseen_class_ids]  # add 1 for background in the first dimension
                bg_seen_class_ids = [i for i in range(len(self.adj_matrix)) if i not in unseen_class_ids]
                new_order = bg_seen_class_ids+unseen_class_ids
                self.adj_matrix = self.adj_matrix[np.ix_(new_order, new_order)]
            # self.adj_matrix = torch.eye(21).float().cuda()
        # self.adj_matrix[-3][-7] = 0

        # self.adj_matrix = torch.tensor(np.load('./data/xview/sim_matrix_xview.npy')).float().cuda() + torch.eye(61).cuda()
        self.pos_thr = pos_thr
        self.loss_scale = loss_scale
        self.margin_scale = margin_scale

    def diversity_regularizing_loss(self, embeddings, firstone=False):
        # Normalize each embedding to have unit norm
        embeddings_norm = embeddings / embeddings.norm(dim=1)[:, None]

        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1))

        # Create an identity matrix of size CxC
        identity = torch.eye(similarity_matrix.size(0)).to(similarity_matrix.device)

        # Subtract identity matrix (which will subtract 1 from the diagonal elements, making them 0)
        similarity_matrix = similarity_matrix - identity

        # Use absolute values, so that both -1 (opposite direction) and 1 (same direction) would be considered high similarity.
        similarity_matrix = torch.abs(similarity_matrix)

        if firstone:
            similarity_matrix = similarity_matrix[0]
            loss = similarity_matrix.sum() / (embeddings.size(0) - 1.)
        else:
        # Sum the absolute values in similarity_matrix and divide by C*(C-1) to get the mean
        # The division factor is due to the fact that we have considered each pair twice.
            loss = similarity_matrix.sum() / (embeddings.size(0) * (embeddings.size(0) - 1))

        return loss

    def extract_group_features(self, embeddings, group_ids):
        unique_group_ids = np.unique(group_ids)
        group_features = [[] for _ in range(len(unique_group_ids))]

        for i, group_id in enumerate(group_ids):
            group_features[group_id].append(embeddings[i])

        return group_features


    def forward(self, features, labels):

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        #
        # batch_size = features.shape[0]
        # labels = labels.contiguous().view(-1, 1)
        #
        # labels = self.group_labels
        # # features = features[labels[:,0]!=0]
        # # labels = labels[labels[:,0]!=0]
        #
        # #cls embedding contrast loss
        # labels = self.group_labels.view(-1, 1)
        #
        # #todo
        # # mask = torch.eq(labels, labels.T).float().to(device)
        # mask = torch.eq(labels, labels.t()).float().to(device)
        # # mask = torch.eq(labels, labels.transpose(1, 0)).float().to(device)
        # #
        #
        # ##todo
        # # anchor_dot_contrast = torch.div(
        # #     torch.matmul(features, features.T),
        # #     self.temperature)
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(features, features.transpose(1, 0)),
        #     self.temperature)
        # ##
        #
        # # normalize the logits for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        #
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size).view(-1, 1).to(device),
        #     0
        # )
        # mask = mask * logits_mask
        # single_samples = (mask.sum(1) == 0).float()
        #
        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log((exp_logits+self.eps).sum(1, keepdim=True))
        #
        # # compute mean of log-likelihood over positive
        # # invoid to devide the zero
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)
        #
        # # loss
        # # filter those single sample
        # loss = - mean_log_prob_pos*(1-single_samples)
        # loss = loss.sum()/(loss.shape[0]-single_samples.sum())


        # group_feats = self.extract_group_features(features, self.group_labels.cpu())
        # withingroup_loss = 0
        # only_unseen_cls = True
        # if only_unseen_cls:
        #     group_feats = group_feats[1:5]
        # for id, feats in enumerate(group_feats):
        #     if len(feats)>1:
        #         withingroup_loss += self.diversity_regularizing_loss(torch.stack(feats, dim=0), firstone=only_unseen_cls)
        # loss+= withingroup_loss

        def triplet_loss_vectorized(embeddings, similarity_matrix, threshold):
            C, D = embeddings.shape

            # Filtering similarity matrix using threshold
            positives_mask = similarity_matrix > threshold
            negatives_mask = similarity_matrix <= threshold

            # For each class, sample one positive and one negative class
            positive_samples = torch.multinomial(positives_mask.float(), 1).squeeze()
            negative_samples = torch.multinomial(negatives_mask.float(), 1).squeeze()

            # Get embedding vectors for anchors, positives, and negatives
            anchors = embeddings
            positives = embeddings[positive_samples]
            negatives = embeddings[negative_samples]

            # Calculate margins (similarity of positive pairs minus similarity of negative pairs)
            margins = (similarity_matrix[torch.arange(C), positive_samples] -
                       similarity_matrix[torch.arange(C), negative_samples])
            margins*=self.margin_scale

            # Calculate triplet loss for all triplets, add margin, and apply ReLU
            losses = F.relu((anchors - positives).pow(2).sum(dim=1) -
                            (anchors - negatives).pow(2).sum(dim=1) + margins)

            # Sum over all classes to obtain the final loss
            loss = losses.sum()

            return loss

        loss_trip = triplet_loss_vectorized(features, self.adj_matrix, self.pos_thr)

        return loss_trip*self.loss_scale