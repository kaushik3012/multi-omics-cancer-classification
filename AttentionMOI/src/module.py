import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMOI(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DeepMOI, self).__init__()
        self.lin1 = nn.Linear(in_feat, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, out_feat)
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = F.dropout(x, 0.5)
        
        x = self.lin2(x)
        x = torch.relu(x)
        x = F.dropout(x, 0.5)
        
        x = self.lin3(x)
        
        return x



class Net(nn.Module):
    def __init__(self, dim_dna, dim_rna, dim_out):
        
        super().__init__()
        self.dna = nn.Sequential(
            nn.BatchNorm1d(num_features=dim_dna),
            nn.TransformerEncoderLayer(d_model=dim_dna, nhead=1, dim_feedforward=128),
        )
        
        self.rna = nn.Sequential(
            nn.BatchNorm1d(num_features=dim_rna),
            nn.TransformerEncoderLayer(d_model=dim_rna, nhead=1, dim_feedforward=128),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_dna + dim_rna, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, dim_out)
        )
        self.norm = nn.BatchNorm1d(num_features=dim_dna + dim_rna)

    def forward(self, data_dna, data_rna):
        feat_dna = self.dna(data_dna)
        feat_rna = self.rna(data_rna)
        h = torch.cat([feat_dna, feat_rna], dim=1)
        h = self.norm(h)
        h = h.squeeze(-1)
        out = self.mlp(h)
        return out
