---
title: Guassian_Splatting 学习
categories: 学术
tags:
  - 计算机视觉
  - 3D渲染
mathjax: true
abbrlink: 6b773604
date: 2024-10-12 02:06:22
---

提出`Guassian-Splatting`的原因正是为了解决【三维重建问题】

[论文地址](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf)   [项目地址](https://github.com/graphdeco-inria/gaussian-splatting)

### 什么是三维重建问题？

三维重建技术是指利用二维投影或影像恢复物体三维讯息（形状等）的数学过程和计算机技术。

因为三维的形状比二维影像有更多资讯，因此直接从二维影像推测三维形状对计算机而言并不是容易的工作，但对人类来说，对于生活中常见到的物体，我们常常可以从单一角度看，就可以推测经验上物体的整个形状(在三维空间中的样子)，人类之所以能做到这样是因为长久经验的累积，因此对于计算机而言，如果学习过够多二维影像以及三维形状的资讯，理论上也能够做到，因此近来有不少利用深度学习的三维重建方法，利用大量的训练资料(例如:影像以及相对应的三维形状)来训练深度神经网络或是卷积神经网络建构的模型，达到由单张或多张二维影像作为输入，推测三维形状。此外，有些大量三维模型的数据库也在近年被建立，以便于这些深度学习模型的训练与相互比较的基础。

输入的内容我们也有所不同：

*不同类型的输入二维资料类型:
- 使用二维RGB影像作为输入
- 使用深度图做为输入

*不同数量的输入影像:
- 使用同一物件的单张影像作为输入
- 使用同一物件的多张影像(由不同视角所拍摄)作为输入


### 方法介绍！

论文中提及的`3dgs`是一项非常厉害的技术，在详细介绍之前我们先有一个笼统的认知、3dgs的过程可以简要概括为以下几步：

- 获取一个3D场景中不同角度的一组照片or一段视频，使用`SFM`等技术初始化这个3D场景的点云阵列、或者干脆直接初始化一个随机的点云
- 将这些点云描述为3维空间中的高斯椭球分布（也就是说每个点除了他自身之外，还应有一个协方差描述他的形状，一个球谐系数来描述他的颜色，以及一个不透明度
- 将这些椭球体沿着特定的角度投影到对应位姿所在的投影平面上，这一步也叫`splatting`，一个椭球体投影到平面上会得到一个椭圆（代码实现时其实是以长轴为直径的圆），然后通过计算待求解像素和椭圆中心的距离，我们可以得到不透明度（离的越近，说明越不透明）。每个椭球体又各自代表自己的颜色，这是距离无关的。于是就可以进行alpha compositing，来合成颜色。然后快速的对所有像素做这样的计算，这被称作”快速可微光栅化“。

<center>
<image src=https://picx.zhimg.com/80/v2-7cbe3b0c3b67ce80593fad0d73a814b5_720w.webp width=40%></image>
</center>

- 光栅化之后便得到了从3d点云得到的一张图片，我们将这张图片和`ground truth`的图片计算损失，反向传播更新参数便可以得到优化的点云阵列

#### 那么问题来了为什么要使用 Guassian 分布呢？

在实际的应用中我们往往需要离散化连续分布，比如三角面，我们只会记录它的三个顶点。
当投影完成后，我们只能做一些有限的操作来阻止“锯齿”，例如对结果进行一个模糊操作，这些操作一般都是局部的。我们这样做的目的，本质是“希望用离散的表达来重建原来的信号，进一步在重建好的信号上进行“resampling”。如果我们对处理后的结果，视觉上看起来没什么混叠或者锯齿上的问题，那就说明“resampling”是成功的。

为了理解在这个过程中，高斯分布为什么重要，我们需要牵扯到信号与系统中的概念。


补充 必要的数字信号处理知识

采样定理的形象化描述：在时域对信号进行采样，等效为在频域对信号频谱进行周期延拓。在频域对频谱进行采样，等效为在时域对信号进行周期延拓。


我们考虑一个采样率为$f_{s}=\frac{1}{T}$，那么一个连续信号$x_{a}(t)$的离散采样可以表述为：

$$
\hat{x}_{a}(t)=\Sigma_{n=-\infty}^{+\infty}x_{a}(t)\delta(t-nT)
$$
（注意：时域上离散的信号，频域上是周期的。这样的采样过程就类似于我们使用3D高斯分布、三角面元这样的操作对3D世界也是相当于一种采样

总所周知，时域采样等价于频域延拓，直接给出对应的频谱：

$$
\hat{X}_{a}(j\Omega)=\frac{1}{T}\Sigma_{n=-\infty}^{+\infty}X_{a}(j\Omega-jn\frac{2\pi}{T})
$$





