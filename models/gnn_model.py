
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

class GNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
