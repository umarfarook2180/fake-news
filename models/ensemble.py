
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, logits_t, logits_g):
        pt = F.softmax(logits_t, dim=1)
        pg = F.softmax(logits_g, dim=1)
        return self.alpha * pt + (1 - self.alpha) * pg
