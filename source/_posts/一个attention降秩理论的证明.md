---
title: 一个attention降秩理论的证明
mathjax: true
categories: 学术
tags:
  - 数学理论
swiper_index: 3
abbrlink: d9fbe1c8
date: 2024-11-09 00:00:40
---


# Attention is not all you need？ 纯粹的注意力机制有什么问题？

[论文地址-> pure attention loses rank doubly exponentially with depth <-](https://arxiv.org/abs/2103.03404)

## 简介

原文链接如上所示，论文开门见山的提出来一个新观点：纯粹的注意力机制会导致矩阵的秩指数级别的下降；论文标题也很有意思`Attention is not all you need`，则是与`LLM`的开山之作`Attention is all you need`相呼应，这篇文章看似在挑战`attention`机制，实际上是在从一个全新的角度来阐述为什么`attention`为什么会表现优异。


## 回忆一下`multi-head attention`机制的细节：


一个**通俗且不严谨**的科普（~~为了不懂NLP的观众~~）：在自然语言处理过程中，我们将每个`word`编码为一个`vector`（我们认为这个向量几何意义上会反映单词的语义信息，你可以理解为比如`原神`和`崩坏铁道`的向量表示相对距离更近、而和`明日方舟`更远，因为后者非米哈游产），从而单词组成的句子就会变成一个`matrix`。自然语言处理中有很多模块负责理解并处理这些`matrix`.

<center>
<img src="/pics/mhaaa.png" width="65%">
</center>


上图就是一个`多头注意力机制`的原理图示。我们先尝试从数学的角度建模这个模块（~~真的很好理解、初中数学水平~~）：




我们考虑一个输入$\mathcal{X}$是一个形如$n\times d_{in}$的输入。那么我们
第h个注意力头的输出可以描述为：

$$SA_{h}(\mathcal{X})=P_{h}\mathcal{X}W_{V,h}+1b_{V,h}^{T}$$


其中，$W_{V,h}$是形如$d_{in}\times d_{v}$的`value`矩阵，$P_{h}$是：

$$
\begin{align}
P_{h} &= \text{softmax}\left(d_{qk}^{-\frac{1}{2}} \left(\mathcal{X}W_{Q,h} + 1b_{Q,h}^{T}\right) \left(\mathcal{X}W_{K,h} + 1b_{K,h}^{T}\right)^{T}\right) \\
&= \text{softmax}\left(d_{qk}^{-\frac{1}{2}} \left(\mathcal{X}W_{QK,h}\mathcal{X}^{T} + 1b_{Q,h}^{T}W_{K,h}^{T}\mathcal{X}^{T}\right)\right)
\end{align}
$$

注意：这里的softmax操作是对矩阵的每一行进行的，$W_{Q,h}、W_{K,h}$的形状都是$d_{in}\times d_{qk}$，于是最后的输出是$n\times n$的形状，并且根据softmax的运算性质每行加上相同的值不会影响最终的输出，所以上述$P_{h}$还可以接着作上述第二个等号的化简。如果你对这其中的某些步骤存在疑问可以关注后续会出一篇深度学习入门的博客文章。


最后我们将多个头的注意力加权合并便得到最终这一层attention的输出：

$$
\begin{align}
SA(\mathcal{X}) & =\sum_{h\in{[H]}}SA_h(\mathcal{X})\\
&=1[b_{O,1}^T,\dots,b^{T}_{O,H}]+[SA_{1}(\mathcal{X}),\dots,SA_H(\mathcal{X})][W_{O,1}^T,\dots,W_{O,H}^T]^T\\
&=\sum_{h\in[H]}P_h\mathcal{X}W_h+1b_{O}^T
\end{align}
$$

其中，$W_h=W_{V,h}W_{O,h}^T$;


我们先忽略上面的偏置项$b_{O}$，那我们一个由多层纯注意力层堆积而成的神经网络的最终输出可以描述为：

$$
\begin{align}
\mathcal{X}^L &=\sum_{h\in [H_L]}P_{h}^{L}\mathcal{X}^{L-1}W_{h}^L\\
&=\sum_{h\in [H_L]}P_{h}^{L}(\sum_{h^\prime\in[H_{L-1}]} P_{h^\prime}^{L-1} \mathcal{X}^{L-2} W_{h^\prime}^{L-1})W_h^{L}\\
&=\sum_{h_1,\dots,h_{L}\in[H]^L}(P_{h_L}^L\dots P^1_{h_1})\mathcal{X}(W_{h_1}^1\dots W_{h_L}^L)
\end{align}
$$

其实形象地，我们不难发现上述式子展开后的每一项都对应着多层注意力网络的一条可行路径（见下图。


<center><img src="/pics/att_path.png" width="75%"></center>


相信看完上述的描述之后，你肯定对线性LLM流行的多头注意力机制有了一个较为细致的了解了吧（~~不确信~~


## pure attention collapse rank 现象？

注意力~~降智~~降秩机制其实描述的是这样的事情：随着大模型层数的增加，如果我们简单的使用注意力层的堆叠，那么最后面的输出矩阵$\mathcal{X}^L$每行的向量**指数级别的倾向于一致，也就是矩阵被降秩了**！！！这对于LLM来说是一个非常糟糕的现象，~~毕竟谁都不希望看到自己的Chatbot只会说"啊对对对对对、啊错错错错错错错"吧~~。后面两个小节我们会分别从数学上证明这种现象和提出这种现象的解决方法$\dots$

## Mathematics Proof of Rank-Collapsing in pure ATTETION

<center>
<span style="text-decoration: line-through; color: red;">终于来到喜闻乐见的数学拷打时间了</span>
</center>



首先我们需要先定义一个残差，来衡量一个矩阵和秩①矩阵的相似程度，我们定义的残差如下：

$$
res(\mathcal{X})=\mathcal{X}-1x^T，where \quad x=argmin_{x}\|\mathcal{X}-1x^T \|
$$

不难验证，一个矩阵如果越越接近于秩①矩阵的话残差是越小的。并且从残差的定义来看（$x$的任意性），偏置项$b_O$是不会影响残差大小的。

### 先来看单个头的单层注意力的情况

对于单个头的一层注意力

$$
\mathcal{X}^{\prime}=SA(x)=P\mathcal{X}W_V
$$

我们先来证明如下结论：

$$
\|res(SA(\mathcal{X}))\|_{1,\infty}\leq 
\dfrac{4\gamma\|W_{QK}\|_{1} \|W_V\|_{1,\infty}}{\sqrt{d_{qk}}}\|res(\mathcal{X})\|_{1,\infty}^{3}
$$

其中$\gamma$是一个常量；

由之前`(2)`式子的推导我们有：

$$
P(\mathcal{X})=\text{softmax}\left(d_{qk}^{-\frac{1}{2}} \left(\mathcal{X}W_{QK,h}\mathcal{X}^{T} + 1b_{Q,h}^{T}W_{K,h}^{T}\mathcal{X}^{T}\right)\right)
$$

我们引入记号$\mathcal{A},\mathcal{R},\mathcal{R^{\prime}}$，其中：

$$
\begin{align}
\mathcal{A}&=\mathcal{X}W_{QK}\mathcal{X}^T+1b_{QK}^{T}\mathcal{X}^T\\
\mathcal{R} &:= res(\mathcal{X})\\
\mathcal{R^{\prime}}&:=res(\mathcal{X}^{\prime})
\end{align}
$$

从而我们的注意力矩阵$\mathcal{A}$可以改写为：

$$
\begin{align}
\mathcal{A}&=(1x^T+\mathcal{R})\dfrac{W_{QK}}{\sqrt{d_{qk}}}(1x^T+\mathcal{R})^T+1b_{QK}^{T}(1x^T+\mathcal{R})^T\\
&=(1x^T\dfrac{W_{QK}}{\sqrt{d_{qk}}}x+\mathcal{R}\dfrac{W_{QK}}{\sqrt{d_{qk}}}x+1b_{QK}^{T}x)1^T + (1x^T+\mathcal{R})\dfrac{W_{QK}}{\sqrt{d_{qk}}} + 1b_{QK}^T\mathcal{R}^T\\
\end{align}
$$

我们再一次使用$\mathcal{softmax}$的平移不变的运算特性，可以得到：

$$
\begin{align}
P&=softmax(\mathcal{R}\dfrac{W_{QK}}{\sqrt{d_{qk}}}\mathcal{R}^T+1r^T)\\
r&:=\mathcal{R}\dfrac{W_{QK}^T}{\sqrt{d_{qk}}}x + \mathcal{R}\dfrac{b_{QK}}{\sqrt{d_{qk}}}

\end{align}
$$



我们设$\mathcal{E}:=\mathcal{R}\dfrac{W_{QK}}{\sqrt{d_{qk}}}\mathcal{R}^T$、$\tilde{A}=1r^T$, 那么我们有：

$$
\begin{align}
P\mathcal{X}&=P(1x^T+\mathcal{R})\\
&=1x^T+P\mathcal{R}\quad\text{（这一步使用了softmax一行加和等于1的性质）}\\
&=1x^T+softmax(\mathcal{E}+1r^T)\mathcal{R}\\
&\leq 1x^T+(I+2D)1sofmax(r)^T\mathcal{R}\quad\text{（操蛋，这一步我没太看懂😤}\\
&=1(x^T+softmax(r)^T\mathcal{R})+2D1softmax(r)^T\mathcal{R}\\
\end{align}
$$

$D矩阵的相关附录见`Appendix`的part1；

~~😤😤😤不等式那一步我也没太看懂作者的意图，D是啥东西作者也没提，矩阵直接比较大小好像就是每个ij位置的元素对应比较。先硬着头皮看下去罢😤😤😤~~

从而我们有：

$$
\|[SA(\mathcal{X})-1(r^\prime)^T]_{ij}  \| \leq 2 \| [ D1softmax(r)^TRW_{V}  ]_{ij}           \|
$$

在此处$r^\prime=(x+\mathcal{R}^{T}softmax(r))W_{V}$;我们再来寻找上述不等式右边的界，考虑$\mathcal{L_1}$范数我们有：

$$
\| [ D1softmax(r)^TRW_{V}  ]_{ij} \|\leq \|D1\|_1 \|\mathcal{R}\|_1\|W_V\|_1
$$

在上述步骤中我们使用了$\|softmax(r)\|_1=1$以及$\|AB\|_1 \leq\|A\|_1\|B\|_1$的性质。从而不难得到$\|[SA(\mathcal{X})-1(r^\prime)^T]\|_1\leq2\|D1\|_1 \|\mathcal{R}\|_1\|W_V\|_1$的结论。

通过类似的分析过程我们同样可以得到$\|[SA(\mathcal{X})-1(r^\prime)^T]\|_\infty\leq2\|D1\|_\infty \|\mathcal{R}\|_\infty\|W_V\|_\infty$.


结合上述两步推导过程我们有：

$$
\|\mathcal{R}^\prime\|_{1,\infty}=\sqrt{\|\mathcal{R}^\prime\|_1\|\mathcal{R}^\prime\|_{\infty}}\leq2\sqrt{\|D1\|_1 \|D1\|_\infty}\|\mathcal{R}\|_{1,\infty}\|W_{V}\|_{1,\infty}
$$








## Appendix

### Lemma-1

引理1：设$P$是矩阵$A$的`row-stochastic matrix`，$\tilde{A}$是矩阵$\tilde{A}=A-E$的`row-stochastic matrix`.(for some matrix $E$ with $\|E_{ij}-E_{ij}^\prime\|\leq 1.256$)，有：


$$
(I-D)\tilde{P}\leq P\leq (I+2D)\tilde{P}
$$

成立，其中对角矩阵$D$满足$D_{ii}=max_{j,j^\prime}\|\delta_i^TE(\delta_j-\delta_{j^\prime})\|$.（这里的$\delta_i$我猜测就是就是第i个元素为1的向量...





<center>剪切线，启动！</center>
------------------------

## 后记


<center>
这一篇论文过于理论化，Mr.Xau抽空前前后后一个星期才看完；内容太抽象导致本人也变得抽象起来了belike：
<img src="/pics/chouxiang.jpg" width="45%">
</center>