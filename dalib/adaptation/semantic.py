from ast import Num
from pydoc import classname
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['SemanticLoss']


class SemanticLoss(nn.Module):
    def __init__(self, n_class: Optional[int] = -1):
        super(SemanticLoss, self).__init__()
        self.n_class = n_class
        self.decay = 0.3
        self.cudable = torch.cuda.is_available()
        self.s_centroid = torch.zeros(self.n_class, 256)
        self.t_centroid = torch.zeros(self.n_class, 256)
        self.s_classvars = torch.zeros(self.n_class)
        self.t_classvars = torch.zeros(self.n_class)
        if self.cudable:
            self.s_centroid = self.s_centroid.cuda()
            self.t_centroid = self.t_centroid.cuda()
            self.s_classvars = self.s_classvars.cuda()
            self.t_classvars = self.t_classvars.cuda()
        self.CEloss, self.MSEloss, self.BCEloss = nn.CrossEntropyLoss(), nn.MSELoss(), nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, s_feature: torch.Tensor, t_feature: torch.Tensor, \
        y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        n, d = s_feature.shape
        
        s_labels, t_labels = torch.max(y_s, 1)[1], torch.max(y_t, 1)[1]
        s_logits, t_logits = torch.max(y_s, 1)[0], torch.max(y_t, 1)[0]

        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        zeros = torch.zeros(self.n_class, d)
        if self.cudable:
            zeros = zeros.cuda()
        
        threshold = 0.9
        zero_s = torch.zeros_like(s_logits)
        zero_t = torch.zeros_like(t_logits)
        s_logits_selected = torch.where(s_logits > threshold, s_logits, zero_s)
        t_logits_selected = torch.where(t_logits > threshold, t_logits, zero_t)  

        s_sum_feature_weighted = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), torch.transpose(s_logits_selected.repeat(d, 1), 1, 0) * s_feature)
        t_sum_feature_weighted = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), torch.transpose(t_logits_selected.repeat(d, 1), 1, 0) * t_feature)

        current_s_centroid_weighted = torch.div(s_sum_feature_weighted, s_n_classes.view(self.n_class, 1))
        current_t_centroid_weighted = torch.div(t_sum_feature_weighted, t_n_classes.view(self.n_class, 1))

        decay = self.decay

        s_centroid = (1-decay) * self.s_centroid + decay * current_s_centroid_weighted
        t_centroid = (1-decay) * self.t_centroid + decay * current_t_centroid_weighted

        semantic_loss = self.MSEloss(s_centroid, t_centroid)

        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()


        return semantic_loss
