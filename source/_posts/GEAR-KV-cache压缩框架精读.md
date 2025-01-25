---
title: 'GEAR:KV cache压缩框架精读'
mathjax: true
categories: 学术
tags:
  - 大模型压缩
abbrlink: 9b786805
date: 2025-01-24 22:29:55
---


# GEAR: 一種高效的近乎無損推理的LLM的 KV cache 壓縮策略

論文地址在這裡[GEAR: An Effective KV Cache Compression Recipe  for Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527)




## 附錄

(a)理解`低秩矩陣`和`稀疏矩陣`

`低秩矩陣`和`稀疏矩陣` 的相同点在于都表明矩阵的信息冗余比较大.具体来说，稀疏意味着有很多零，即可以压缩；低秩意味着矩阵有很多行（列）是线性相关的.low rank matrix和稀疏矩陣各有各的用途.

- 補充知識點1 稀疏表示!

假設有$m$維的n個輸入$X=[x_1,x_2,...,x_n]$,m非常之大,我們想要一勞永逸,不想要存儲所有的m個向量,於是我們想要學習到一種表示:

$$
\begin{align*}
<D,Z>&=argmin_{D,E}\|X-DZ \| \\
s.t.\|Z_i\|_{0} &\leq \Delta \\
\end{align*}
$$

這裡的$D$我們稱做字典,$Z$是輸入基於字典的稀疏表示法;$\Delta$是一個較小的值,第二個式子能夠保證$Z$矩陣是一個稀疏矩陣.尋求上式的最優解是一個NP-Hard問題,我們可以使用一些算法來快速地得到一些次優解.

針對上述問題,我們可以先選取一組稀疏表示的初始解:
$D_0,Z_0$,其中$Z_0$是滿足上述第二個不等式約束的.

然後的優化目標變成了:

$$
minimize \quad \|X-DZ\|+\lambda \sum_{i} \|{X_0}_i\|
$$


這裡存在$D,Z$兩個優化變量,一般的策略就是固定其中一個變量然後動態更新另外一個變量,下面以更新字典$D$為例子.

假設我們固定了稀疏表示$Z$,在此基礎上來逐列更新字典的第k列$d_k$:

$$
\begin{align*}
\|X-DZ\|&=\|X-\sum_{j=1}^{K}d_j\cdot z_j\|\\
&=\|(X-\sum_{i\neq k}d_j\cdot x_j)-d_k\cdot z_k\|\\
&=\|E_k-d_k\cdot z_k\|
\end{align*}
$$


上式中$E_k=(X-\sum_{i\neq k}d_j\cdot x_j)$被定義為殘差;此時最優化問題可以被描述為$min_{d_k}\|E_k-d_k\cdot z_k\|$,這顯然是一個最小二乘問題,可以直接用最小二乘法就可以解決這個問題.

但是這裡仍然需要註意的問題是,我們不能直接使用$E_k$進行求解,因為不加限製的求解時$x_k$不能保證稀疏性.我們需要選取出$E_k$中不為0的部分再進行迭代更新.就像下圖所展示的一樣:

<center>
  <img src="/pics/zerop.png" width="80%">
</center>

參考資料[稀疏表示](https://www.cnblogs.com/endlesscoding/p/10090866.html)

- 補充知識點2 低秩(low rank represent ~~低排名🐻‍❄️~~)表示!


假設一個輸入信號$X$由低秩矩陣$R$和噪聲$S$組成,即$X=R+S$,为了還原低秩矩阵，求解如下最小化问题：

$$
<D,Z>=arg min(rank(R))+\lambda\|S\|_{l=0},s.t.X=R+S;
$$

然而,矩陣rank的計算和L0范數通常是非凸的,考慮到這點我們通常使用矩陣的核范數 $\|\cdot\|*$(矩陣奇異值的和) 和L1范數 $\|\cdot\|_{1}$  對上式進行鬆弛處理,

$$
<D,Z>=arg min(\|R\|*)+\lambda\|S\|_{l=1},s.t.X=R+S;
$$

從而得到一個凸優化問題.


(b)稀疏子空間聚類(Sparse Subspace Clustering, SSC)

稀疏子空間聚類問題(SCC)可以描述為:
假設有一組高維數據點的集合$X=[x_1,x_2,...,x_n]$,其中$x_i\in R^D$,是高維空間中的點(这些点分布在 K KK 个低维子空间上，每个子空间的维数远小于数据点的原始维度，即 d_k << D ).
對於這一組數據我們期望尋找一組`稀疏向量`$Z=[z_1,z_2,...,z_n]$,
使得$x_i$能夠被其他數據點的線性組合來逼近.



和上面低秩表示一樣我們定義一個鬆弛化的SCC最小化函數

$$
minimize\quad \|x_i-X\cdot z_i\|_2+ \lambda \|z_i\|_1
$$

上式意味著我們在整體數據集上為每一個數據點尋找一個盡可能稀疏的表述法則,從而將數據進行聚類.


<center>
  <img src="/pics/katsumi.jpg" width="80%">
</center>




