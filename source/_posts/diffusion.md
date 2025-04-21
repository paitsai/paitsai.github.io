---
title: 最近想和大家讲讲`diffusion model`!
mathjax: true
categories: 学术
tags:
  - 生成式模型
  - diffusion model
abbrlink: aa10f68a
date: 2025-04-19 02:11:47
---


<!-- # 最近想和大家讲讲`diffusion model`! -->


# `Diffusion`和图像生成的关系

谈到`diffusion model`那么就不得不谈及`AIGC`. 在过去几年里里，以Stable Diffusion为代表的AI图像/视频生成是世界上最为火热的AI方向之一. Stable Diffusion里的这个”Diffusion”是什么意思？其实，扩散模型(Diffusion Model)正是Stable Diffusion中负责生成图像的模型。想要理解Stable Diffusion的原理，就一定绕不过扩散模型的学习。


在这篇文章里，我会由浅入深地对去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）进行一个介绍。


# 图像生成任务的解决

相比其他AI任务，图像生成任务显然是一个更加困难的事情. 比如人脸识别,序列预测...这一系列任务都有明确的训练集来给出or蕴含一个[标准答案].
但是图像生成就没有, 图像生成数据集里只有一些同类型图片，却没有指导AI如何画得更好的信息。

过去的解决方案:

## `GAN`对抗生成模型


- `GAN`的原理简介

GAN的主要结构，包括一个生成器G（Generator）和一个判别器D（Discriminator），整个训练过程，便是二者的对抗博弈：
给定参考数据集$p_{data}(x)$, 希望学习出$G,D$使得最优化下面的函数:

$$
min_{G}max_{D}V(D,G)=\mathcal{E}_{x\sim P_{data}}\left[Log D(x)\right]+\mathcal{E}_{z\sim p(z)}\left[Log(1-D(G(z)))\right]
$$

它的含义其实就是: 对于生成模型$G$, 输入是随机噪声 $z\sim p_{z}(z)$, 输出为 $G(z)$ , 上面第二项就是使 $G(z)$ 越能够迷惑判别器越好. 判别器 $D(x)$ 输入真实数据 or $G(z)$ 判别器需要对两者进行辨别.

<center>
<img src="/pics/GAN.png" width="80%">
</center>

- `GAN`存在的问题:

(*) 无法用于解决`离散型数据`的生成问题, 自然语言处理是一个很典型的例子:

局部信息很重要：图像局部很多细节并不太影响人类的对图像的理解，只要整体到位就 ok，不然也犯不着 CNN 这么多 filter 一层层给你过滤，你破坏少数像素点不影响人类理解。`自然语言麻烦在于，在细微处修改一下`，就变味了。比如“西瓜汁好喝！”，我稍微改一下“西瓜汁好喝吗？”，尾巴动一点，整个意思都变了。GAN 局部信息重构到底是靠死记硬背训练样本，还是靠神经网络插值“生成”出来的？我反正不清楚，不管如何，针对自然语言这种细节敏感的问题，GAN 不是一个首选方案，不然 n-gram 的 LM 也不会活到今天。

- 解决办法(引入强化学习RL)

related works [SeqGAN](https://arxiv.org/pdf/1609.05473v5)

to be continued


------------------------------------------------------------------------------


## `VAE` (Variational AutoEncoder) 变分推断模型

VAE作为可以和GAN比肩的生成模型，融合了贝叶斯方法和深度学习的优势，拥有优雅的数学基础和简单易懂的架构以及令人满意的性能，其能提取disentangled latent variable的特性也使得它比一般的生成模型具有更广泛的意义。

- 关于`Latent Variable`(隐藏变量)的理解

生成模型一般会生成多个种类的数据，比如说在手写数字生成中，我们总共有10个类别的数字要生成，这个时候latent variable model就是一个很好的选择。

为什么呢？举例来说，我们很容易能注意到相同类别的数据在不同维度之间是有依赖存在的，比如生成数字5的时候，如果左边已经生成了数字5的左半部分，那么右半部分就几乎可以确定是5的另一半了。


<center>
<img src="/pics/VAE.png" width="80%">
</center>

因此一个好的想法是，生成模型在生成数字的时候有两个步骤，即(1)决定要生成什么数字，这个数字用一个被称为latent variable的向量z来表示，(2)然后再根据z来直接生成相应的数字。用数学表达式来表示就是：

$$
P(X)=\int P(X|z;\theta)P(z)dz.
$$


- 问:那么现在的关键是关于`Latent Variable` $z$ 的 先验概率分布形式 $P(z)$ 如何取值?


答:很简单,直接设定 $P(z)$ 满足`标准高斯分布`就行. 因为任何复杂的分布都可以通过多层MLP映射成标准高斯分布.

- 问: 如何训练一个VAE

答: 最大化 $P(X)=\int P(X|z;\theta)P(z)dz$ 即可;

(1) 有了$z$的先验分布知识,我们可以使用若干次采样来最大化`似然函数`


即最大化 $P(X)\approx =\dfrac{1}{n}\sum_{i}P(X|z_i)$

然而当$z$是维度很高的高斯分布的时候,这种方法训练十分**低效**. 直接使用`z`先验分布来训练低效的原因直觉上是很明显的.因为对于数据集中的一个实例 $X_j$ 而言,其对应的隐变量区间 $z_j$ 实际上被似然函数**采样到的概率是很低的**.也就是说有效的训练次数很低.我们需要先假设一个 $q(z|X)$ 从此来针对数据集 $X_j$ 先得到 $z_j$ 来针对`decoder`训练,这样有效训练次数将大幅提升!





或者换一种说法: 我们需要注意到, 对于采样 $z_i \sim P(z)$ 所有的 $P(X|z_i)$ 其实都是几乎为0的. 换言之,绝大部分采样得到的的 $z$ 对于目标函数 $P(X)$ 的贡献**无足轻重**. 在换言之,我们只需要关注 $P(z|X)$ 更大的部分即可.

那么问题来了, 怎么计算 $z$ 的后验知识 $P(z|X)$?????? ~~很难的!~~


<center>
<img src="/pics/anon_red.jpg" width="35%">
</center>



(2) 贝叶斯公式巧妙转换 $p(z|X)$

直接得到后验分布 $P(z|X)$ 是极其困难的,我们能够得到的只有`encoder`侧的输出 $q(z|X)$ .我们需要记`encoder`的输出 $q(z|X)$;但是与此同时必须保证 $p$ 和 $q$ 的**分布相似性**.这里用`KL`散度来衡量:

$$
D(p(z|X)\|q(z|X))=E_{z\sim q}\left[log(q(z|X))-log(p(z|X))\right]
$$

使用**贝叶斯公式**对上式化简~~化繁~~: (~~其实贝叶斯这一步是最关键的一步~~)

$$
p(z|X)=\dfrac{p(X|z)\times p(z)}{p(X)}
$$

> 可以看见:我们通过使用贝叶斯公式将 $p(z|X)$ **巧妙地转换**为 $p(X|z)$ 将问题从`encoder`一侧转移到`decoder`一侧 ! 这是最最关键的一步!

于是:


$$
\begin{align*}
D(p(z|X)\|q(z|X))&=E_{z\sim q}\left[log(q(z|X))-log(p(z|X))\right]\\
&=E_{z\sim q}\left[log(q(z|X))-log(\dfrac{p(X|z)\times p(z)}{p(X)})\right]\\
&=E_{z\sim q}\left[log(q(z|X))-log(p(X|z))-log(p(z))+log(p(x))\right]
\end{align*}
$$

再度化简可以得到 $\rightarrow$

$$
log(p(X))-KL\left[q(z|X)\|p(z|X)\right]=E_{z\sim q}\left[log(p(X|z))\right]-KL\left[q(z|X)\|p(z)\right]
$$

注意到`KL`散度的非负性,于是有:

$$
log(p(X)) \geq E_{z\sim q}\left[log(p(X|z))\right]-KL\left[q(z|X)\|p(z)\right]
$$

我们不妨记作:

$$
ELBO=E_{z\sim q}\left[log(p(X|z))\right]-KL\left[q(z|X)\|p(z)\right]
$$

`ELBO`(Variational Lower Bound)记作变分下界;至此,我们近似将问题转化为了最大化变分下界;
既然目标是让变分下界最大化，那么我们就需要仔细研究一下这个变分下界。

- 首先是第一项，要想最大化 ELBO，那我们自然是想让第一项尽可能的大，也就是 x given z 的概率分布期望值更大。这很明显就是由 z 到 x 重组的过程，也就是 AutoEncoder 中的 Decoder，从潜在空间 Z 中重组 x。模型想做的是尽可能准确地重组.

- 其次是第二项，要想最大化 ELBO，我们自然需要让这项 KL 散度尽可能小，也就是 潜在空间 z 的近似后验分布尽可能接近于 z 的先验分布！这一项我们可以理解为，模型想让 z 尽可能避免过拟合.



--------------------------------------------------------------------------------


# `Diffusion`模型


扩散模型是一种特殊的VAE，其灵感来自于热力学：一个分布可以通过不断地添加噪声变成另一个分布。放到图像生成任务里，就是来自训练集的图像可以通过不断添加噪声变成符合标准正态分布的图像。但是:

(1) 不再训练一个可学习的编码器，而是把编码过程固定成不断添加噪声的过程；

(2) 不再把图像压缩成更短的向量，而是自始至终都对一个等大的图像做操作。解码器依然是一个可学习的神经网络，它的目的也同样是实现编码的逆操作。


<center>
<img src="/pics/DM.png" width="90%">
</center>


具体来说，扩散模型由正向过程和反向过程这两部分组成，对应VAE中的编码和解码。在正向过程中，输入 $X_0$ 会不断混入高斯噪声. 经过 $T$ 回合的加噪处理之后, 图像 $X_T$ 会变成一个符合标准正态分布的纯噪声图像. 而在反向过程中，我们希望训练出一个神经网络，该网络能够学会若干个去噪声操作，把 $X_T$ 还原回 $X_0$ .


## PART1 加噪过程:

前向加噪过程可以用描述为:

$$
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t \mathcal{I})\\
$$

$$
q(x_{1:T}|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1})=\prod_{t=1}^{T}\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t \mathcal{I})\\
$$

其中 $\beta_{t_i}$  是高斯分布方差的超参数,在扩散过程中，随着 $T$ 的增大, 越来越接近纯噪声。当 $T$ 足够大的时候，收敛为标准高斯噪声 $\mathcal{N}(0,\mathcal{I})$。

不妨设 $\alpha_t=1-\beta_t$ , $\hat{a_t}=\prod_{i=1}^t \alpha_i$ , 依次展开 $x_t$ 可以得到:

$$
\begin{align*}
x_t&=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_1\\
&=\sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_2\right)+\sqrt{1-\alpha_t}\epsilon_2\\
&=\sqrt{\alpha_1\alpha_2}x_{t-2}+\left(\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_2+\sqrt{1-\alpha_t}\epsilon_1\right)\\
\end{align*}
$$

其中 $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0,\mathcal{I})$, 由独立正态分布的可叠加性: $\mathcal{N}(0,\sigma_1^2\mathcal{I})+\mathcal{N}(0,\sigma_2^2\mathcal{I})$ :

$$
x_t=\sqrt{\alpha_t\alpha_{t-1}}x_{t-1}+\sqrt{1-\alpha_t\alpha_{t-1}}\hat{\epsilon}
$$

再进一步:

$$
x_t=\sqrt{\hat{a_t}}x_0+\sqrt{1-\hat{a_t}}\hat{\epsilon_t}
$$



这意味着 $q(x_t|x_0)= \mathcal{N}(x_t|\sqrt{\hat{a_t}}x_0,(1-\sqrt{\hat{a_t}})\mathcal{I})$

加噪过程到此结束.


## PART2  解噪过程

实际上, 每一步降噪过程 $q(x_{t-1}|x_t)$ 是难以形式化求解的.  我们的解码器就是为此而来的!其中 $\theta$ 就是我们神经网络的参数:

$$
p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1}|\mu_{\theta}(x_t,t),\sigma_{\theta}^2(x_t,t)\mathcal{I})
$$

于是有:

$$
p_{\theta}(x_;T)=p(x_T)\prod_{t=T}^{1}p_{\theta}(x_{t-1},x_0)=p(x_T)\prod_{t=T}^{1}\mathcal{N}(x_{t-1}|\mu_{\theta}(x_t,t),\sigma_{\theta}^2(x_t,t)\mathcal{I})\\
$$




--------------------------------------------------------------------------------


<center>
<img src="/pics/mtm_layer.gif" width="40%">
</center>

