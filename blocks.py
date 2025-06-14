import torch
import torch.nn as nn
import torch.nn.functional as Fn


class FeedForward(nn.Module):
    def __init__(self, inFeatures, hiddenFeatures, outFeatures, drop_rate = 0.3):
        super().__init__()
        self.inFeatures = inFeatures
        self.hiddenFeatures = hiddenFeatures
        self.outFeatures = outFeatures
        self.linear1 = nn.Linear(inFeatures, hiddenFeatures)
        self.linear2 = nn.Linear(hiddenFeatures, outFeatures)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        x = self.dropout(Fn.gelu(self.linear1(x)))
        x = self.dropout(Fn.gelu(self.linear2(x)))
        return x
    



