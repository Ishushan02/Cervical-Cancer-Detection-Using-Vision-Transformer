# Vision-Transformer
A simple Vision Transformer (VIT) implementation for image classification tasks. This repo showcases the ViT architecture, using self-attention to process image patches. It includes model training, evaluation, and performance analysis, aimed at understanding ViT’s application in computer vision.

## Patch Embedding
Image is splitted into blocks of m * n, and then each patch is seperated, Those patches are then feed forwarded to 
Neural Network to get its embedding. All channels(R, G, B) of the patches are stacked together 1 after the other.

[Pixel1r, Pixel1g, Pixel1b, Pixel2r, Pixel2g, Pixel2b, ....]

The Fully Connected Neural Network outputs Number of Patches * D ; where D is the dimension of  Attention. Now we have embedding of all the patches, hence we have to infuse it with Positional Embedding. At position 0 we add CLS token and rest places we add those to positional embedding using Sinusodial Formula.
So, basically what Patch Embedding does is
    - seperates the images into patches
    - add positional embedding
    - add CLS token at the begining (for SUmmary of the Task)
