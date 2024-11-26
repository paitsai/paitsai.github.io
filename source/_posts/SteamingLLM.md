---
title: SteamingLLM
mathjax: true
categories: 学术
tags:
  - 大模型加速
  - KV Cache 优化
abbrlink: '17847744'
date: 2024-11-26 18:54:02
---
~~只会开新坑不会填旧坑的屑Mr.Xau🕊️🕊️🕊️~~ 又打算分享一篇基于KV Cache优化经典工作

~~本人近期任务太多以后一定会填坑的（下次一定~~
🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺🥺

# StreamLLM 基于注意力汇聚现象的KV Cache策略

[论文地址](https://arxiv.org/abs/2309.17453)，Streaming这个词很有意思，~不是我们玩的steam~ ，它可以是喷射的意思，也可以是涓涓细流的意思；我觉得从这篇工作的内容来看，翻译为**娟娟溪流**可能更加合适一点。那么这个娟娟细流到底指的是LLM中的什么，但是正所谓`铁打的衙门流水的官`,KV Cache中有没有铁打不变的东西呢？且听后文分析。

## 知识补充

主要是KV Cache的介绍：什么是KV Cache，为什么只缓存KV？

### 什么是KV Cache？


回忆一下Transformer中的注意力机制，在经典的Transformer中我们有向量化的语料输入`Tokens序列`$\mathcal{X}$，如果batch size=1，$\mathcal{X}$ 的形状是$[l,d_h]$ ，其中$l$是输入序列的长度 ~一句话单词数~ ， $d_h$是给每个单词的编码向量的维度 ;经过注意力编码之后有（暂时以一个头的注意力为例子）：

$$
\begin{align*}
\mathcal{Q}&=W_Q\cdot \mathcal{X}\\
\mathcal{K}&=W_K\cdot \mathcal{X}\\
\mathcal{V}&=W_V\cdot \mathcal{X}\\
\end{align*}
$$

我们通过这个编码之后的$\mathcal{QKV}$矩阵，可以计算输出的注意力分数：

$$
\mathcal{Att_s}=\mathcal{QK}^T
$$

我们把$\mathcal{Att_s}=\mathcal{QK}^T$展开写看看：

$$
\begin{align*}
\mathcal{Att_s}&=\mathcal{QK}^T\\
&=\begin{bmatrix}
q_1  \\
q_2  \\
\vdots \\
q_l
\end{bmatrix}\cdot [k_1,k_2,\cdots k_l]\\
&=\begin{bmatrix}
q_{1}\cdot k_{1} & q_{1}\cdot k_{2} & \cdots & q_{1}\cdot k_{l} \\
q_{2}\cdot k_{1} & q_{2}\cdot k_{2} & \cdots & q_{2}\cdot k_{l} \\
\vdots & \vdots & \ddots & \vdots \\
q_{l}\cdot k_{1} & q_{l}\cdot k_{2} & \cdots & q_{l}\cdot k_{l}
\end{bmatrix}
\end{align*}
$$


如果采用仅解码器的架构，由于掩码的存在，会有：

$$
\begin{align*}
\mathcal{Att_s}=\begin{bmatrix}
q_{1}\cdot k_{1} & 0 & \cdots & 0 \\
q_{2}\cdot k_{1} & q_{2}\cdot k_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
q_{l}\cdot k_{1} & q_{l}\cdot k_{2} & \cdots & q_{l}\cdot k_{l}
\end{bmatrix}
\end{align*}
$$

我们都知道每次LLM会将上一次输出的一个token放在下次输入的最后一个，那么下一轮的注意力分数是：

$$
\begin{align*}
\mathcal{Att_s|_{next}}=\begin{bmatrix}
q_{1}\cdot k_{1} & 0 & \cdots & 0 & 0 \\
q_{2}\cdot k_{1} & q_{2}\cdot k_{2} & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots& \vdots \\
q_{l}\cdot k_{1} & q_{l}\cdot k_{2} & \cdots & q_{l}\cdot k_{l} & 0\\
q_{l+1}\cdot k_{1} & q_{l+1}\cdot k_{2} & \cdots & q_{l+1}\cdot k_{l} & q_{l+1}\cdot k_{l+1}\\
\end{bmatrix}
\end{align*}
$$

显然因为掩码的存在，注意力分数变成了一个下三角矩阵，因此当我们产生了新的tokens的时候，只需要计算最新tokens的q查询向量$q_{l+1}$和$k_{k+1}$即可；这样的结构我们很自然就能想到，每次LLM进行推理的时候将之前$1\to k$的$\mathcal{K},\mathcal{V}$都缓存起来，下次计算注意力分数的时候便无须再次重新计算了；

从上面的公式的形状，我们也不难发现，完全没有必要缓存$\mathcal{Q}$矩阵，因为$\mathcal{Q}$每次下三角矩阵扩展的时候都只用到了$q_{1+k}$这一列，所以，之前计算的$\mathcal{Q}$对于后面的计算是没有作用的我们可以完全舍弃以节省显存开销。



### KV Cache有什么问题

随着输入序列的增加，如果缓存所有的$\mathcal{KV}$矩阵，那么对显存的需求将会二次上升。这就要求我们设计高效的KV Cache的动态调整策略。详细的KV Cache大小估算公式可以参考附录。

这篇工作为什么叫StreamLLM，我的理解是因为即使随着对轮对话的进行，上下文越来越长，StreamLLM也能够基于注意力汇聚这一现象合理的关键的初始和一定上文窗口的cache来使得即使上下文增长也能像涓涓细流一样稳定流畅的推理。



-----------------------------

蓑鱿剪切线

-----------------------------



## 附录



to be continue


<center>
<img src="/pics/kefuxiang.jpg" width="60%">
</center>