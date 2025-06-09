# Vision-Transformer
A simple Vision Transformer (VIT) implementation for image classification tasks. This repo showcases the ViT architecture, using attention to process image patches. It includes model training, evaluation, and performance analysis, aimed at understanding ViT’s application in computer vision.

# Dataset
- Execute dataset.py to download the Data 
#### 1. **im\_Metaplastic** 
    : These are changing cells that are **not cancer**, but appear during normal repair or transformation, They're important because cancer often starts in these areas.
    : Not dangerous by themselves, but need to be watched.

#### 2. **im\_Dyskeratotic**

    : These are **abnormal-looking cells that may be a sign of pre-cancer or cancer, they show early signs of damage.
* These are **warning signs** and should be looked at carefully.

#### 3. **im\_Superficial-Intermediate**

    : These are normal, healthy cells** from the outer layer of the cervix, Their presence usually means everything is fine.

#### 4. **im\_Parabasal**

    : These are **immature cells** that are deeper in the tissue, If you see a lot of them, it could mean the tissue is healing or there's a hormonal issue, especially after menopause.
    : Not cancer, but can show other changes.

### 5. **im\_Koilocytotic**

    : These cells show **HPV infection**, which is the virus that can lead to cervical cancer, These are early warning cells — not cancer, but they tell doctors to keep an eye out.

| Cell Type                | What it Means             | Is it Dangerous?         |
| ------------------------ | ------------------------- | ------------------------ |
| Metaplastic              | Changing but not cancer   | Usually not              |
| Dyskeratotic             | Damaged, possibly serious | Yes, watch closely       |
| Superficial-Intermediate | Normal healthy cells      | No                       |
| Parabasal                | Immature, often harmless  | Sometimes                |
| Koilocytotic             | HPV-infected cells        | Can lead to cancer later |


## Patch Embedding
Image is splitted into blocks of m * n, and then each patch is seperated, Those patches are then feed forwarded to 
Neural Network to get its embedding. All channels(R, G, B) of the patches are stacked together 1 after the other.

[Pixel1r, Pixel1g, Pixel1b, Pixel2r, Pixel2g, Pixel2b, ....]

The Fully Connected Neural Network outputs Number of Patches * D ; where D is the dimension of  Attention. Now we have embedding of all the patches, hence we have to infuse it with Positional Embedding. At position 0 we add CLS token and rest places we add those to positional embedding using Sinusodial Formula.
So, basically what Patch Embedding does is
    - seperates the images into patches
    - add positional embedding
    - add CLS token at the begining (for SUmmary of the Task)

