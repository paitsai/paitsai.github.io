---
title: 今天想和大家谈谈`DiT`(Diffusion Transformer)
mathjax: true
categories: 学术
tags:
  - 生成式模型
  - diffusion model
  - transformer
abbrlink: 8b4a5e79
date: 2025-04-23 21:40:44
---


论文地址[click here](https://arxiv.org/pdf/2212.09748)


`DiT`主要基于Vision Transformer（`ViT`）架构，该架构对patches序列进行操作，`DiT`保留了`ViT`的大部分配置。为了将`tranformer`架构用在视觉领域上, 我们需要将图片信息进行序列化操作(**patchify**):


# patches2tokens


原始`diffusion`模型中VAE加噪得到一个隐空间状态记作 $z\sim shape(I,I,C)$, 首先可以考虑使用一个线性嵌入层将`patches`转换为`tokens`序列,如同下图所示:




<center>
<img src="/pics/DiT.png" width="80%">
</center>

最后转换出来的tokens序列的长度由patching size **p**所决定,最终得到tokens序列的长度就是 $(\dfrac{l}{p})^2$ ;


# DiT block architecture

原始ViT中，在patchify之后，输入tokens会直接由一系列Transformer块处理。但DiT的输入除了noise图像外，有时还会处理额外的条件信息，如noise时间步 $t$ ，类标签 $c$ ，输入自然语言等。

<center>
<img src="/pics/DiT_block.png" width="99%">
</center>

# Transformer Decoder Forward

在最后的DiT块之后，使用layer norm（如果使用adaLN，则自适应）及线性解码器将得到的图像tokens序列解码为输出noise和输出对角线协方差，每个token会被线性解码为形如 $(P\times P\times 2C)$  的张量，其中 $X$  是输入空间的通道数。最后，将解码后的tokens重新排列成原始的空间布局，得到预测的noise和协方差。输出的noise和对角线协方差的形状和原始输入的形状相同。



<center>
<img src="/pics/alice.png" width="40%">
</center>
