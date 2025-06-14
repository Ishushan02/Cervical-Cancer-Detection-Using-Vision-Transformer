import torch.nn as nn
import torch
from patchEmbedding import PatchEmbedding
from blocks import FeedForward, TransformerEncoder



class VisionTransformer(nn.Module):

    def __init__(self, imageSize, patchSize, inChannels, numClasses, embedDim, numLayers, numHeads, mlpDimension, drop_rate):
        super().__init__()

        self.path_embedding = PatchEmbedding(imageSize, patchSize, inChannels, embedDim)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(embedDim, numHeads, mlpDimension, drop_rate)
            for _ in range(numLayers)]
        )
        self.layerNorm = nn.LayerNorm(embedDim)
        self.out = nn.Linear(embedDim, numClasses)

    
    def forward(self, x):
        x = self.path_embedding(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.layerNorm(x)
        # print(x.shape)

        cls_token = x[:, 0]
        # print(cls_token.shape)

        out = self.out(cls_token)
        # print(out.shape)

        return out

# test = torch.randn(10, 3, 64, 64)
# vit = VisionTransformer(imageSize=64, patchSize=4, inChannels=3, numClasses=10, embedDim=40, numLayers= 6, numHeads=4, mlpDimension=40, drop_rate=0.2)

# out = vit(test)

# print(out.shape)