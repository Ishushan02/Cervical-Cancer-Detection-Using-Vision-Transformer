import torch
import torch.nn as nn



class PatchEmbedding(nn.Module):
    def __init__(self, imageSize, patchSize, inputChannels, embedDim):
        super().__init__()
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.inputChannels = inputChannels
        self.embedDim = embedDim

        self.patches = (imageSize // patchSize) ** 2
        self.projection = nn.Conv2d(in_channels=inputChannels, out_channels=embedDim, kernel_size=patchSize, stride=patchSize)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedDim)) 
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patches, embedDim))


    
    def forward(self, x):
        batch, channels, height, width = x.shape

        x = self.projection(x)
        print(x.shape)
        x = x.flatten(2).transpose(1, 2) 
        # torch.Size([1, 8, 10, 10]) -> torch.Size([1, 8, 100]) -> torch.Size([1, 100, 8])
        # print(x.shape)
        cls_tokens = self.cls_token.expand(batch, -1, -1) 
        # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  
        # print(x.shape, )
        x = x + self.pos_embed[:, :x.shape[1], :]  
        # print(x.shape)     
        
        return x
    
# # Testing the Embed functions
# test = torch.randn(1, 3, 128, 128)  
# PE = PatchEmbedding(imageSize=256, patchSize=12, inputChannels=3, embedDim=8)
# out = PE(test)
# print(out)
