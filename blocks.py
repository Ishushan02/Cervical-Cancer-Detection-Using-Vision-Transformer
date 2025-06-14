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
    

# x = torch.randn(10, 64, 64)
# ffModel = FeedForward(64, 100, 10)

# out = ffModel(x)
# print(out.shape)


class TransformerEncoder(nn.Module):
    def __init__(self, nEmbed, numHeads, mlpOut, drop_rate = 0.2):
        super().__init__()
        self.nEmbed = nEmbed
        self.numHeads = numHeads

        self.norm1 = nn.LayerNorm(nEmbed)
        self.attention = nn.MultiheadAttention(embed_dim=nEmbed, num_heads=numHeads, dropout=drop_rate, batch_first=True)
        self.mlp = FeedForward(nEmbed, 2 * nEmbed, nEmbed, drop_rate)

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm1(x))
        return x
    
# x = torch.randn(10, 64, 64)  
# att = TransformerEncoder(64, 8, 10)
# out = att(x)
# print(out.shape)  # should be [10, 64, 10] if mlpOut = 10
