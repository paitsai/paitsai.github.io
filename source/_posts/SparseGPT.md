---
title: LLM剪枝-SparseGPT方法
mathjax: true
categories: 学术
tags:
  - 大模型压缩
  - 剪枝
  - Pruning
abbrlink: 20461be3
date: 2024-11-11 22:01:11
---
# 'SparseGPT' one-shot Pruning Strategy

[【论文地址】SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)

## `SparseGPT`简介

`SparseGPT`是一种用于压缩Massive LLM的一次性（one-shot）、无须重训练（no need for retraining）的基于非结构化Pruning（剪枝）的模型压缩方法，发现了至少50%的稀疏性；

- `SparseGPT`提出的创新点何在？其实就是两点：`ONE-SHOT` && `NO RETRAINING`;

<center>
<img src="/pics/retrain.png" width="60%">
</center>

上图就是原来的模型减枝之后，我们仍然需要一个Sparse Retraining的过程来调整稀疏化之后的模型，SparseGPT提出的剪枝方法则是one-shot的，也就是无须后面retraining或者说调整的代价很小。

## `SparseGPT`的基本原理

### 已有方法存在的问题：

一般的模型减枝（Pruning）都包含两步——Mask Selection 和 weight restruction。

假设我们某一层的模型权重记作$W_\mathcal{L}$，输入记作$\mathcal{X}_\mathcal{L}$，掩码矩阵记作$M_\mathcal{L}$，Sparsify之后的权重变成了$\tilde{W_\mathcal{L}}$，那么我们的最优化目标就变成了：

$$
argmin_{M_\mathcal{L},\tilde{W_\mathcal{L}}} \|W_\mathcal{L}\mathcal{X}_\mathcal{L}-(M_\mathcal{L}\bigodot \tilde{W_\mathcal{L}} \mathcal{X}_\mathcal{L})  \|_2^2
$$

但是因为${M_\mathcal{L},\tilde{W_\mathcal{L}}}$这两部分会同时影响到上述最优化的结果，也有证明这是一个NP-hard问题，在巨大的模型面前解决这个问题是不现实的，我们需要另找办法。一种有效的方法就是根据一些显式的法则（比如直接根据权重来筛选）来实现直接给定一个MASK。然后接着做权重重建即可。

根据上述分析和推导，我们的权重重建过程可以化为一个最小二乘法的最优化问题，形式通解可以描述为：

$$
\mathcal{W}^i_{\mathcal{M}_i}|_{update}=(\mathcal{X}_{\mathcal{M}_i}{\mathcal{X}_{\mathcal{M}_i}}^T)^{-1}{\mathcal{X}_{\mathcal{M}_i}}(\mathcal{W}_{\mathcal{M}_i}{\mathcal{X}_{\mathcal{M}_i}})^T\\
$$

我们不妨定义海森矩阵：

$$
H_{\mathcal{M_i}}=\mathcal{X}_\mathcal{\mathcal{M}_i}{\mathcal{X}_\mathcal{\mathcal{M}_i}}^T
$$

这里面的${\mathcal{X}_{\mathcal{M}_i}}$指的是经过掩码的第$i$行之后仍然存在的输入；$\mathcal{W}_{\mathcal{M}_i}$是对应的权重；$\mathcal{W}^i_{\mathcal{M}_i}$是第i行更新之后的权重。

但是这样的方法仍然会存在很多问题：

- **最重要的一点是**：掩码每一行不同会导致不同的海森矩阵，导致计算量巨大。并且$(H_{\mathcal{M_i}})^{-1}\neq(H)^{-1}_{\mathcal{M_i}}$，计算矩阵的逆也十分消耗计算资源，就像下图所展示的这样。

<center>
<img src="/pics/row_hs.png" width="70%">
</center>

---

<center>！素食剪切线警告！</center>

---

### 基于OBS思想的权重重建

> 一种等价的迭代视角

作者借鉴了[Optimal Brain Surgery【OBS】](https://www.babak.caltech.edu/pubs/conferences/00298572.pdf)（相关介绍见附录部分）中调整剩余权重来减少Pruning所减去当前权重影响的思想来对现有的方法进行改进。但是相比较于原来OBS更改全局参数的策略，SparseGPT则是使用一种更加高效的方法，每次只更新未被Pruning掉的部分；

<center>
<img src="/pics/sparseGPT.png" width="80%">
</center>

上图就是 `SparseGPT` Pruning算法的可视化，每次对**一列**权重进行Pruning（白色的块是被减掉的），右侧深蓝色的块就会被参数更新来补偿修建错误。没被修改的块就不会对后续参数造成影响。

工作亮点：

> 1）海森矩阵的处理

我们将输入的特征矩阵来进行编号$j=1,2\dots,d_{col}$，然后我们定义特征矩阵子集$U_j$的简记方式：

$$
U_{j+1}=U_j-{j}; \quad U_1=\{1,2\dots,d_{col}\}
$$

也就是说$U_1$代表的是全集；$U_{j+1}$是在$U_j$的基础上删除序号为$j$的元素.于是有：

$$
(H_{U_j})^{-1}=((\mathcal{X}\mathcal{X}^T)_{U_j})^{-1}
$$

根据[这篇论文](https://arxiv.org/abs/2208.11580)的工作，我们选取依次少选取一行的优势就显示出来了，我们计算海森矩阵的逆可以根据上一步的逆很快的得到。

设$B=(H_{U_j})^{-1}$，那么在$\mathcal{O}(d_{col}^2)$的时间之内我们可以计算出来：

$$
(H_{U_{j+1}})^{-1}=(B-\dfrac{1}{[B]_{1,1}}\cdot B_{:,1}B_{1,:})_{2:,2:}
$$

相比于原来$\mathcal{O}(d_{col}^3)$的复杂度来计算一个矩阵的逆，现在充分利用已有信息可以在$\mathcal{O}(d_{col}^2)$得到答案；

> 2）计算复杂度分析

通过上述的分析我们可以看到整体的计算开销主要由三部分组成：

- （a）初始Hessian矩阵的计算$T_1=\mathcal{O}(n\cdot d_{col}^2)$，其中n是输入特征向量的数量；
- （b）计算初始Hessian矩阵的逆$T_2=\mathcal{O}(d_{col}^3)$；
- （c）然后对每一行使用权重重建$T_3=\mathcal{O}(d_{col}^2d_{row})$

总结：总共的时间复杂度就是$\mathcal{O}(d_{col}^3+d_{row}d_{col}^2)$.对于Transformer系列的模型，可以简化为$\mathcal{O}(h_{hidden}^3)$的复杂度。

### 自适应掩码选择

在此之前，我们主要集中于谈论权重重建的细节，都是基于一个固定的Pruning Mask来进行的权重重建。已有的Mask Selection方法可以参考基于幅度（magnitude）选取的方法；一个最直观地做法就是选取每一列值最小的$p\%$的权重，这样可以直接构造出$p\%$的稀疏性，但是这样对每一行来说是不平均的，特别是transformer这样的架构会有少量高敏感的权重。

为了解决这一点所存在的问题，原文使用了一种叫做迭代阻塞（iterative blocking）的方法。相比于每次选取一列来做一个$%p$$p\%$的pruning，文中每次选取$B_s=128$来进行一个mask的选择，这样可以一定程度上避免这种pruning不均匀的现象；

同时SparseGPT稀疏化方法同样可以适用于很多半结构化稀疏性，例如很有名的[n:m细粒度结构化稀疏性](https://arxiv.org/abs/2102.04010)，所谓**N:M细粒度结构化稀疏**即在M个连续的权重值中固定有N个非零值，其余元素均为零值，例如下图中N=2，M=4的N:M稀疏（2:4稀疏）。N:M稀疏的权重矩阵可以进行压缩存储，仅保留所有的非零元素，再辅以一个indices矩阵指示非零元素在原矩阵中的空间位置。从实验结果上来看，采用N:M细粒度结构化稀疏方案的模型在权重存储量和计算量大减的同时取得了比肩甚至超过稠密的原网络的推理精度。

<center><img src="/pics/mnspar.png" width="80%"></center>

至此我们可以就更具给定的$m,n$来确定$B_s=m$，来完成Mask Selection的工作。

整体的算法流程可以描述如下：

<center>
<img src="/pics/spgptal.png" width="60%">
</center>

### 实验验证部分

- 硬件条件：单块80G A100显卡
- 模型：BLOOM-176B 、OPT-175B系列模型（都是decoder-only的架构）
- 衡量指标：Perplexity困惑度【见附录】
- 测试数据集：raw-WiKiText2、PTB、C4
- Sparsify时间：4h左右

> SparseGPT和幅度剪枝效果对比


可以发现，SparseGPT算法能够提供更多的稀疏性：

<center>
<img src="/pics/1731671407844.png" width="60%">
</center>


<center>
<img src="/pics/1731672468503.png" width="60%">
</center>


实验结果表明：相比于规模性对较小的视觉任务的模型，LLM在50%的sparsity的设定下，幅度减枝（magnitude pruning）效果下降更加明显。但是对于SparseGPT而言，这种趋势相对平缓许多，对于2.7B的模型，Perplexity的下降已经接近一个百分点，到175B的模型时，下降已经接近于0；


<center>
<img src="/pics/1731673090354.png" width="80%">
</center>

另外对于模型不同部位稀疏化影响敏感性的研究：

**later layers are more sensitive than earlier ones**

<center>
<img src="/pics/1731674061016.png" width="60%">
</center>


## Appendix

### 结构化剪枝VS非结构化剪枝技术

常见的稀疏方式可分为结构化稀疏和非结构化稀疏。前者在某个**特定维度（特征通道、卷积核等等）**上对卷积、矩阵乘法做剪枝操作，然后生成一个**更小的模型结构**，这样可以复用已有的卷积、矩阵乘计算，无需特殊实现推理算子；**后者以每一个参数为单元稀疏化**，然而并不会改变参数矩阵的形状，只是变成了含有大量零值的稀疏矩阵，所以更依赖于推理库、硬件对于稀疏后矩阵运算的加速能力。

半结构化剪枝是一种介于结构化剪枝和非结构化剪枝之间的剪枝方法，可以同时实现高精度和结构正则化。半结构化剪枝通常基于特定的模式进行剪枝，这些模式可以是任意的，但需要经过精心设计以减轻性能下降并实现特定的加速效果。**半结构化剪枝可以被视为一种细粒度的结构化剪枝方法。**

[参考](https://cloud.tencent.com/developer/article/2370076)阅读资料

### OBS：‘optimal brain surgery’ 的介绍：

OBS这个名字很有意思，翻译过来就是最佳脑外科手术的意思；很贴合OBS对神经网络所做的事情：他将神经网络中一些不重要的权重、连接给切除之后，再对其他权重做调整来最小化减去神经的损失，OBS这个名字非常切合。

我们考虑对神经网络的误差函数进行泰勒展开可以都得到：

$$
\delta \mathcal{E}=(\dfrac{\delta\mathcal{E}}{\delta \mathcal{W}})^T\cdot \delta\mathcal{W}+\dfrac{1}{2}{\delta\mathcal{W}}^TH\delta\mathcal{W}+\mathcal{O}(\|\delta\mathcal{W}\|^3)
$$

其中$\mathcal{H}$为Hessian矩阵，$\mathcal{W}$代表的模型当前的权重参数，$\mathcal{E}$代表训练误差。训练神经网络用任意的优化算法，该剪枝算法都是适用的。

我们可以通过一些梯度下降优化算法来找到一个局部最小解，上述公式第一项就等于0，再忽略三阶无穷小，可以得到：

$$
\delta\mathcal{E}=\dfrac{1}{2}\delta\mathcal{W}^T H\delta\mathcal{W}
$$

这个时候我们开始对较小的权重进行减枝，比如我需要剪切权重中的第$\mathcal{q}$个元素那么我可以这样描述：

$$
e_q^T\delta \mathcal{W}+\mathcal{W}_q=0
$$

其中$e_q$为单位向量，只有在第$q$项为1，其余项皆为0；这意味着什么？这意味着进行下一步更新的时候$\mathcal{W}_q$将被直接置于0！相当于完成了剪切；

由上述的推导，我们现在将问题转化为了一个带约束条件的最优化问题，写出拉格朗日方程：

$$
L=\dfrac{1}{2}\delta\mathcal{W}^T H\delta\mathcal{W}+\lambda (e_q^T\delta \mathcal{W}+\mathcal{W}_q)
$$

解这个拉格朗日方程我们可以得到：

$$
\delta\mathcal{W}=-\frac{\mathcal{W}_q}{[H^{-1}]_{qq}}\cdot H_{:,q}^{-1};\quad \delta L_q=\frac{\mathcal{W}_q^2}{2[H^{-1}]}_{qq}
$$

### Perplexity in LLM

困惑度在LLM中的定义是：

$$
PP(X)=exp(-\frac{1}{t}\sum_{i=1}^t logp(x_i|x_{j<i})))
$$

也就是说，在LLM中困惑度（Perplexity）被定义为一个token概率序列的指数平均负对数似然估计。

困惑度越高说明每次产生下一个单词时候选词概率越平均或者说候选词更多。如果用困惑度来衡量一个LLM的表现，那么自然是困惑度越低LLM的表现越好。




---

<center>素食剪切线</center>

---

## 后记

这篇文章真的很厉害，特别是每次OBS更新只选取部分子集进行更新，从而大量简化逆矩阵的运算量真的让人印象深刻！impressing！！！

<center><img src="https://drive.miyago9267.com/d/file/img/mygo/%E9%80%99%E5%82%A2%E4%BC%99%E6%A0%B9%E6%9C%AC%E4%BB%80%E9%BA%BC%E4%B9%9F%E4%B8%8D%E6%87%82.jpg" width="75%"></center>

什么时候我才能写出这样的文章😶‍🌫️😶‍🌫️😶‍🌫️

参考资料：

- https://www.cnblogs.com/sasasatori/p/17809829.html