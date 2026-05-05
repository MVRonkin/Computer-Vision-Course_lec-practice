![](06Transformer_174.jpg)

![](06Transformer_175.png)

__Architectures__  __Transformers__

_Main architectures of_  _convolutional_  _ neural networks\._

_Course: Computer Vision\._

Ronkin Mikhail Vladimirovich

PhD in Engineering, Associate Professor of IRIT-RTF, UrFU

# Limitations of convolutional networks

Image Processing

\- Manual formalization of features and decision-making

Very limited tasks, up to 100 images

\- Manual formalization of features and automatic decision-making

Limited tasks, up to 1000 images

\- Automatic formalization of features and decision-making

Limited pre-training

Interpretability, Formalization

Limited tasks, up to 1 million images

\- Automatic solution based on special pre-training

Several tasks, up to 100 million images and beyond

Foundation model

\- zero-shot pre-training

Several tasks, up to 100 million images and beyond

Model complexity / abstractness of tasks and their breadth and quantity, Data volume

# Properties and Limitations of convolutional networks

| Property | Goal | Limitation |
| :-: | :-: | :-: |
| __Locality of processing__ | allows accounting for proximity relations between neighboring pixels | May make it difficult to understand the global context of the image. Fixed filter size is not always optimal for capturing the object |
| __Invariance of processing__ | The filter acts identically regardless of scale, position, rotation of the object | for a number of tasks etc. this is a minus, it's important that a smile means corners of the mouth go up not down (see the idea of capsule networks). |
| __Receptive field is formed through depth linearly__ | To "see" distant parts of the image, CNN must accumulate the receptive field | receptive field through pooling eats up small local details – for objdet/semseg tasks etc. this is a minus. depth of receptive field turns out to be limited |
| __Small number of parameters in a layer__ | Simplification of the learning task on small data | Reduces network capacity during pre-training on very large datasets (e.g. JFT-300M/JFT-3B) – limits the quality |
| __Inductive bias__  __of images__ | Intuitively clear why convolution works for narrow image processing tasks | Convolution is not a universal architecture, so for a wide set of CV and multimodal tasks Inductive bias does not work |

https://habr\.com/ru/articles/922868/

https://habr\.com/ru/articles/935726/

# Visual Transformer (VIT)

__Splitting the image into __  __patches__  __: __ instead of processing the entire image as a whole, ViT divides it into small fragments (patches), for example, 16x16 pixels in size\.

__Converting __  __patches__  __ into __  __embeddings__ : each patch is converted into a vector of fixed length\.

__Adding positional encoding__ : since transformers do not take into account the order of input data, information about the position of each patch is added\.

__The resulting sequence is processed by a standard __  __transformer__ : the model learns to determine which parts of the image are most important for performing a specific task

![](06Transformer_176.png)

"An Image is Worth 16x16 Words" \(Dosovitskiy et al\.\, 2020\)

https://github\.com/lucidrains/vit\-pytorch

https://habr\.com/ru/search/?target\_type=posts&order=relevance&q=\[vision\+transformer\]

# Transformer architectures - VIT



* __ViT __  __architecture__  - rethinking of the BERT architecture for natural language processing\.
  * __At the core of the architecture is the multi-head__  __ __  __self-attention__  __ layer\.__
  * _The model is fully _  _fully-connected_  _\._
* The model showed an accuracy of 88\.5% by top-1 method on the ImageNet dataset
  * The model is pre-trained on the JFT-300M dataset (a set with 300 million images)\.
    * In the case of pre-training on small datasets (less than 14 million images), the accuracy of the ViT model decreases significantly (to about 75%)\.
    * The number of parameters of ViT is 10 times higher than for a convolutional network (~ 600 million parameters)\.


Transformer block

# VIT. STEAM

![](06Transformer_179.png)

![](06Transformer_180.png)

https://cs231n\.stanford\.edu/slides/2025/lecture\_8\.pdf

https://habr\.com/ru/articles/935726/

# VIT. CLS token

![](06Transformer_181.png)

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer/

https://habr\.com/ru/companies/otus/articles/849756/

![](06Transformer_182.png)

The output embedding of the \[cls\] token is fed into the Classification Head for making the final decision\.

During training, the model learns to put knowledge about the entire image into the embedding of this token\.

There can also be other tokens, for their own tasks

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer/

https://habr\.com/ru/companies/otus/articles/849756/

# VIT. Positional Encoding

![](06Transformer_183.png)

Positional encodings help the model differentiate between different positions in the image and capture spatial relationships\.

Final dimensionality:  __\[197\, 768\]__  \(196 patches \+ 1 class token\)

https://habr\.com/ru/articles/935726/



* In CV, position is not as critical as in texts –
  * The context of images is much less sequential than text\!
* Therefore, learnable encoding parameters can be introduced\.
  * Essentially, the encoding has a mechanism like the cls token
  * resistance to flip, rotate, crop, etc\. is required
    * All this blurs the encoding


![](06Transformer_184.png)

![](06Transformer_185.png)

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer/

https://arxiv\.org/pdf/2010\.11929

# VIT. STEAM

![](06Transformer_186.png)

__At the output of __  __steam__  __ some set of __  __patches__  __ is obtained – __  __tokens__  __ with__  __ text\-__  __alignment__

https://cs231n\.stanford\.edu/slides/2025/lecture\_8\.pdf\#page=5\.00

# VIT. Transformer blocks

![](06Transformer_187.png)

![](06Transformer_188.png)

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer/

<span style="color:#d63384"> </span>  <span style="color:#d63384">pre</span>  <span style="color:#d63384">\-LN</span>  <span style="color:#1d1d1d"> — approach \(</span>  <span style="color:#1d1d1d">bottom</span>  <span style="color:#1d1d1d">\) works better, </span>  <span style="color:#1d1d1d">than </span>  <span style="color:#d63384">post</span>  <span style="color:#d63384">\-LN</span>  <span style="color:#1d1d1d"> \(top\). And in fact </span>  <span style="color:#d63384">pre</span>  <span style="color:#d63384">\-LN</span>  <span style="color:#1d1d1d"> has become the standard\.</span>

https://d2l\.ai/chapter\_attention\-mechanisms\-and\-transformers/vision\-transformer\.html

https://stackoverflow\.com/questions/70065235/understanding\-torch\-nn\-layernorm\-in\-nlp

Encoder block:

├── Layer Normalization

├── Multi\-Head Self\-Attention

├── Residual Connection

├── Layer Normalization

├── Feed\-Forward Network

└── Residual Connection

![](06Transformer_189.png)

In the original ViT paper, self\-attention has Dropout layers,

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer/

https://arxiv\.org/pdf/2303\.13731

https://habr\.com/ru/articles/935726/

![](06Transformer_190.png)

![](06Transformer_191.png)

In fact, you can get Q, W, K with one projection \+ split or 1\-d convolutions

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer/

https://habr\.com/ru/articles/925050/

![](06Transformer_192.png)

https://arxiv\.org/pdf/2303\.13731

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer

__The head part is often formed from __  __\[__  __cls__  __\]__  __ __  __patch__  __\.__

_Sometimes the average of _  _patches_

![](06Transformer_193.png)

__Note about the number of __  __patches__  __ N\.__  Usually ViT is pre\-trained on large datasets, and then fine\-tuned on relatively small datasets\. It is often useful to fine\-tune at a higher resolution compared to the resolution during pre\-training\.

https://arxiv\.org/pdf/2303\.13731

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer

# Advantages of transformers



    * __Global attention __ \- building complex dependencies between features regardless of their mutual location and from the very first layer\.
  * __Advantages in __  __pre-training__  __/scaling \- __ Transformers will allow training models with significantly increased number of parameters\. This makes it possible to train on large datasets such as JFT\-300M / JFT\-3B\.
      * __Way to improve accuracy and generalization ability through __  __pre-training__ \.
      * __Self\-supervised __  __pretraining__  __ __ \- Possibility for some architectures to pre\-train on unlabeled data\.
        * Pre\-training without labeling contributes to increasing the expressive power (complexity of features) and generalization ability of the neural network\.
  * __Multimodality__  – transformers can effectively work with any modalities together due to their universality
  * __Parallelization__  __ – __ the transformer block is even better adapted to the GPU/TPU structure


https://habr\.com/ru/articles/591779/

Disadvantages of transformer architectures:

for effective training of ViT  __large amounts of data are needed__ \. Otherwise the model is prone to overfitting\.

__Computational resources__ : Transformers have many more parameters than CNN\.

__VIT __  __ has no intuition for images__ \.  \- On simple tasks CNN is better

it is better to fine\-tune at higher resolutions than pre\-training\.

_Mean_   _Attention_   _Distance_  is an analog of  _receptive_   _field_ \.

https://habr\.com/ru/articles/591779/

https://habr\.com/ru/articles/599057/

# Replacement for the receptive field



* The maximum "receptive field" is achieved instantly — each pixel already in the first layer theoretically "sees" all other pixels\.
  * Attention heads in ViT use global information\.
  * Some heads learn locally — for example, paying attention only to the nearest patches\. This is especially noticeable in early layers\.


_C_  _NN_  _ _  _receptive field_

![](06Transformer_194.png)

_Mean_   _Attention_   _Distance_  is an analog of  _receptive_   _field_ \.

That is, how much different features are highlighted between heads

https://cs231n\.github\.io/understanding\-cnn/

https://theaisummer\.com/vision\-transformer/

For a well\-trained CNN \- often demonstrates good and smooth filters

Note that the weights of the first level are very pleasant and smooth, which indicates good convergence of the network\. The color and grayscale controls are grouped because AlexNet contains two separate processing streams, and the obvious consequence of this architecture is that one stream processes high\-frequency grayscale controls, and the other \- low\-frequency color elements\." ~ Stanford University course on CS231: Visualization

Thus, the author has shown that early layer representations can have similar properties\.



* The "receptive field" of ViT is a flexible, learnable attention mechanism that can be either local or global — even in the first layer\.
* ViT flexibly combines local and global interactions from the very beginning
* The receptive field in ViT is not a fixed area, but a learnable mechanism
  * Some heads have small "attention distance" → local filters\.
  * Others — global, covering the entire image even in the first layer\.


![](06Transformer_195.png)

_Mean_   _Attention_   _Distance_  is an analog of  _receptive_   _field_ \.

https://cs231n\.github\.io/understanding\-cnn/

https://theaisummer\.com/vision\-transformer/

For a well\-trained CNN \- often demonstrates good and smooth filters

Note that the weights of the first level are very pleasant and smooth, which indicates good convergence of the network\. The color and grayscale controls are grouped because AlexNet contains two separate processing streams, and the obvious consequence of this architecture is that one stream processes high\-frequency grayscale controls, and the other \- low\-frequency color elements\." ~ Stanford University course on CS231: Visualization

Thus, the author has shown that early layer representations can have similar properties\.

__In C__  __NN__  __ __ as you go deeper into the network (taking into account stride, pooling, etc\.), the receptive field grows locally and sequentially, often linearly (depending on the architecture)\.

Early CNN layers see only local information — edges, textures, simple patterns\.

_VIT_  _ abandons local convolutions_  _ in favor of global _  _self\-attention_  _, which allows the model to directly model dependencies between any parts of the image — and this is what became the key to a new leap in performance on large datasets\._

![](06Transformer_196.png)

Key property of self\-attention \- Any pixel can directly interact with any other pixel already in the first layer\.

The receptive field of one pixel in the first ViT layer is the entire image patch

_VIT_  _ in the first layers gives less smooth _  _features_  _, but more diverse – worse network convergence, but higher capacity_

https://cs231n\.github\.io/understanding\-cnn/

https://theaisummer\.com/vision\-transformer/

For a well\-trained CNN \- often demonstrates good and smooth filters

Note that the weights of the first level are very pleasant and smooth, which indicates good convergence of the network\. The color and grayscale controls are grouped because AlexNet contains two separate processing streams, and the obvious consequence of this architecture is that one stream processes high\-frequency grayscale controls, and the other \- low\-frequency color elements\." ~ Stanford University course on CS231: Visualization

Thus, the author has shown that early layer representations can have similar properties\.

# ViT – a wide family of models

![](06Transformer_197.png)

https://github\.com/lucidrains/vit\-pytorch

![](06Transformer_198.png)

Problem – in the average value, the patch stores its position

or you need to add position encoding after each layer

Or remove it from normalization\.

RMS Norm \- No mean subtraction, and no bias \- only normalization

Dark patch (low brightness) → low RMS → amplification\.

Bright patch → (high brightness) = → high RMS → attenuation\. Token contrast, without losing offset

![](06Transformer_199.png)

![](06Transformer_200.png)

![](06Transformer_201.png)

https://arxiv\.org/pdf/2303\.13731

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer

![](06Transformer_202.png)

Problem – in the average value, the patch stores its position

or you need to add position encoding after each layer

Or remove it from normalization\.

RMS Norm \- No mean subtraction, and no bias \- only normalization

Dark patch (low brightness) → low RMS → amplification\.

Bright patch → (high brightness) = → high RMS → attenuation\. Token contrast, without losing offset

https://rohitbandaru\.github\.io/blog/Transformer\-Design\-Guide\-Pt2/

https://habr\.com/ru/articles/926368/

https://jerryxio\.ng/posts/nd\-rope/

https://arxiv\.org/pdf/2403\.13298

https://arxiv\.org/pdf/2303\.13731

https://blog\.deepschool\.ru/cv/vit\-vision\-transformer

Not all features are equally useful for each patch\.

GatedMLP allows the model to dynamically disable irrelevant dimensions — like soft attention, but inside a single token _»\._

![](06Transformer_203.png)

https://uxlfoundation\.github\.io/oneDNN/dev\_guide\_graph\_gated\_mlp\.html

![](06Transformer_204.png)

![](06Transformer_205.png)

_Not all features are equally useful for each _  _patch_  _\. _  _GatedMLP_  _ allows the model to dynamically disable irrelevant dimensions — like _  _soft_  _ _  _attention_  _, but inside a single _  _token_  _»\._

https://uxlfoundation\.github\.io/oneDNN/dev\_guide\_graph\_gated\_mlp\.html

![](06Transformer_206.png)

Essentially, a mixture of experts is a system with a decision-making committee

V\-MoE replaces a subset of the dense feedforward layers in  _[ViT](https://sh-tsang.medium.com/review-vision-transformer-vit-406568603de0)_  with sparse MoE layers\, where  __each image patch is "routed" to a subset of "experts" \(MLPs\)__ \.

![](06Transformer_207.png)

![](06Transformer_208.png)

![](06Transformer_209.png)

![](06Transformer_210.png)

https://apxml\.com/courses/mixture\-of\-experts\-advanced\-implementation/chapter\-5\-integrating\-moe\-into\-architectures/moe\-in\-vision\-transformers

https://sh\-tsang\.medium\.com/review\-scaling\-vision\-with\-sparse\-mixture\-of\-experts\-dd4de8ad27fa

Essentially, a mixture of experts is a system with a decision-making committee

![](06Transformer_211.png)

V\-MoE replaces a subset of the dense feedforward layers in  _[ViT](https://sh-tsang.medium.com/review-vision-transformer-vit-406568603de0)_  with sparse MoE layers\, where  __each image patch is "routed" to a subset of "experts" \(MLPs\)__ \.

![](06Transformer_212.png)

![](06Transformer_213.png)

![](06Transformer_214.png)

![](06Transformer_215.png)

![](06Transformer_216.png)

https://sh\-tsang\.medium\.com/review\-scaling\-vision\-with\-sparse\-mixture\-of\-experts\-dd4de8ad27fa

MultiHead\(X\,X′\) =  =H1W1O\+H2W2O\+⋯\+HhWhO=h∑i=1HiWiO

Thus, we presented Multi\-Head Attention as a sum of matrices, and no additional restrictions on the properties of WO appeared — these are just two different notations of the same expression\.

Now let's move on to Mixture\-of\-Head Attention\. To do this, replace the sum in the expression above with a weighted sum, that is, each matrix will be multiplied by a scalar gi, which is equal to zero only if the i\-th head is not selected as an "expert":

![](06Transformer_217.png)

https://sebastianraschka\.com/llms\-from\-scratch/ch04/07\_moe/

https://blog\.deepschool\.ru/architecture/moh\-multi\-head\-attention\-as\-mixture\-of\-head\-attention/

# SWIN Transformer

VIT problem – wide patches do not take into account small details \- it is not possible to do  _pixel\-level_  _ prediction\._

_The greater the depth, the less attention to details is needed – more abstract features\._

_Idea \- _  _Hierarchical Vision Transformer using Shifted Windows_  _ \(_  _SWIN\) – _  _each layer should have its own _  _patch_  _ size_

![](06Transformer_218.jpg)

https://habr\.com/ru/articles/599057/

first layer patches 4x4, \- allows processing finer context\. Then 8x8…32x32

_Patch_   _Merging_  concatenates features of neighboring tokens (in a 2x2 window) and reduces dimensionality, obtaining a higher\-level representation\.

each  _Stage_  forms feature "maps" containing information at different spatial scales,

This allows obtaining a hierarchical representation of the image,

![](06Transformer_219.jpg)

https://habr\.com/ru/articles/599057/

_Multi\-_  _Head_   _Attention_  \- quadratic complexity

The idea of (Shifted) Window Multi\-Head Attention \- for each token to compute  _Attention_  not with all other tokens, but only with those within a window of fixed size \( _Window_   _Multi\-Head_   _Attention_ \)\.

![](06Transformer_220.jpg)

If the dimensionality of tokens is  _C_ , and the window size is  _MxM_ , then the complexity for  _\(_  _Window_  _\)_   _Multi\-_  _Head_   _Self_   _Attentions_

![](06Transformer_221.png)

_Attention_  now works in linear time with respect to  _hw_

https://habr\.com/ru/articles/599057/

![](06Transformer_222.jpg)

![](06Transformer_223.jpg)

If the dimensionality of tokens is  _C_ , and the window size is  _MxM_ , then the complexity for  _\(_  _Window_  _\)_   _Multi\-_  _Head_   _Self_   _Attentions_

![](06Transformer_224.png)

_Attention_  now works in linear time with respect to  _hw_

https://habr\.com/ru/articles/599057/

# MOBILE VIT

![](06Transformer_225.png)

https://github\.com/lucidrains/vit\-pytorch/blob/main/images/mbvit\.png

# SWIN Transformer

shifting Attention windows increases their number

implementation of this layer with naive padding of the original feature "map" with zeros will require computing more  _Attentions_  (9 instead of 4 in the example) than we would have computed without shifting

The authors proposed cyclically shifting the image itself before computing and then computing already masked  _Attention_ , to exclude interaction of non\-neighboring tokens\. This approach is computationally more efficient than the naive one, since the number of computed  _Attentions_  does not increase

![](06Transformer_226.jpg)

![](06Transformer_227.jpg)

_Attention_  now works in linear time with respect to  _hw_

https://habr\.com/ru/articles/599057/

https://amaarora\.github\.io/posts/2022\-07\-04\-swintransformerv1\.html

Also in  _Swin_ , the authors used somewhat different  _positional_   _embeddings_ \. They were replaced with a learnable matrix  _B_ , called  _relative_   _position_   _bias_ , which is added to the product of  _query_  and  _key_  under the softmax:

![](06Transformer_228.png)

_Attention_  now works in linear time with respect to  _hw_

https://habr\.com/ru/articles/599057/

# ConvNext

![](06Transformer_229.png)

![](06Transformer_230.png)

![](06Transformer_231.png)

https://habr\.com/ru/companies/otus/articles/654279/
