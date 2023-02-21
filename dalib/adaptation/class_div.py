from ast import Num
from pydoc import classname
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['Class_div']


class Class_div(nn.Module):
    def __init__(self):
        super(Class_div, self).__init__()
        
    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        #diversity loss
        msoftmax = torch.mean(y_t, dim=0)
        ones = torch.ones_like(msoftmax)
        uni_dtb = (1/msoftmax.size(0)) * ones
        class_div = F.kl_div(msoftmax, uni_dtb, reduction='mean')
        return class_div
