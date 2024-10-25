---
title: SE_based_on_unlabled_data
mathjax: true
date: 2024-10-23 23:58:12
categories: 学术
tags:
- ICASSP 
- 音频处理
---

# 无监督音频增强相关技术学习

## pre数学知识

### （1）基础线性代数和高数内容

### （2）广义逆矩阵

> 定义：如果对于一个线性方程组$Ax=b$，如果存在一个矩阵$A^{-}$满足：$\forall b$，$x=A^{-}b$都是原方程的解，那么我们称$A^{-}$为$A$的广义逆矩阵，又叫伪逆矩阵。（伪逆矩阵一般不止一个。

广义逆矩阵意味着什么？如果$A^-$是$A$的广义逆，一条充要条件：    

- 必要性，显然有$AA^-A=A$成立；充分性上，$\forall b$，取$Ay=b$，那么显然有$AA^-Ay=Ay$成立，也就意味着$\forall b$，有$AA^-b=b$成立.


求解广义逆矩阵的通法：

根据非满秩矩阵$A$ 的相抵标准型：$\exist$ 可逆方阵$P、Q$使得

$$
PAQ=\begin{pmatrix} E_r & 0 \\ 0 & 0 \end{pmatrix}
$$

于是：

$$
A=P^{-1}\begin{pmatrix} E_r & 0 \\ 0 & 0 \end{pmatrix} Q^{-1}，
注意M会是A的反形矩阵.
$$

直觉上显然有：

$$
M=Q\begin{pmatrix} E_r & X \\ Y & Z \end{pmatrix}P
$$

会是$A$的广义逆的通解形式！

### （3）一些信号处理的知识

一、形象地，我们可以用波形图来描述一段音频，横轴代表时间、纵轴代表振幅：

![](https://static.emastered.com/images/blog-assets/7093.webp)

但是波形图在频率信息的表征上仍然不够直观，为了弥补这一点引入了频谱图的工作：与波形图一样，频谱图上的时间也沿着 x 轴前进。不同的是，另一个轴代表频谱，低频在底部，一直延伸到人类听觉的最高处。

沿着 y 轴，您将看到构成声音的所有单个频率；基频或根频，使声音具有感知音高，谐波则构成声音的独特色彩和音调。

特定声音的响度由信号的 "热图 "来定义。热图可以用颜色或强度表示，具体取决于您使用的频谱图软件。但从本质上讲，声音越大，它的亮度就越高。吉他手都喜欢这样/doge

![](https://static.emastered.com/images/blog-assets/7084.webp)


二、 $SNR\_loss$ 衡量信号相似度的函数定义：

$$
\begin{equation}
    \begin{cases}
        s_{target}=\frac{<\hat{s},s>s}{ {\mid s\mid }^2 } \\
        e_{noise}=\hat{s}-s_{target}\\
        SNR(s,\hat{s}) =10log \frac{\mid s_{target}\mid ^2}{\mid e_{noise}\mid ^2} 
    \end{cases}
\end{equation}
$$


## 开始介绍一些SE的工作基础（Signal Enhance）

### 无监督音频信号分离技术（Unsupervised Sound Separation）

参考文献：[Unsupervised Sound Separation Using Mixture Invariant Training](https://arxiv.org/pdf/2006.12701)



我们传统的有监督音频分离技术任务，需要训练数据同时包含混合音频和每个音轨单独的音频；这对训练数据的要求很高、导致收集数据集难度很高。基于此提出来一种无监督的训练方法，这种方法只需要混杂の的原始音频即可。训练算法如下：

<center>
![](/pics/mixit.png)
</center>


算法的基本思想是：既然没有原始的音频，那我们不如直接拿来原始混杂的音频，经过一个`DNN`之后，得到期待的分离开的每个音轨的纯净音频，再将这些音频重新组合得到一系列混合音频，从这些混合音频中挑选和原始输入混杂音频相似度最好的几段计算损失函数来反向传播更新参数。