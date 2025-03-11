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


## 研究背景

原文中作者總結了現在階段為了解決GPU Memory問題的流行的幾種方法:

(a)使用`offload`技術,通過將GPU的內存消耗轉移到CPU使用的內存or NVMe的存儲空間上.這種方式對總線帶寬(bandwidth)需求極大

(b)緊接著提出來的是`tokens dropping`技術(比如我們上一篇文章StreamLLM也屬於這一類),這類方法屬於是利用注意力分佈的稀疏性,將註意力分數低下的tokens捨棄達到降低顯存消耗的目的.

(c)另一種經常使用的量化技術(quantization),通過將全精度的數據轉化為半精度的數據進行存儲來降低顯存消耗.

上述的三種方法:(a)會依賴於總線的帶寬來達到GPU和CPU之間高速的數據傳送.(b),(c)兩種方式雖然在絕大部分任務中都能高效的降低顯存佔用,並且對推理效果的損失也極低;但是在復雜的生成式任務中(比如涉及邏輯推理,解決數理問題)這兩種方法都存在普遍且明顯的效能損失.

在較為簡單的任務中,模型只需要產生少數tokens從少數特定的上下文中就可以完整正確的自回歸任務.然而,在複雜的任務中,通常需要模型依據大量相關的上下文tokens產生更長更多的tokens;然而自回歸的docode過程中每一步得會累積誤差;


<center>
  <img src="/pics/approx_err.jpg" width="80%">
</center>

積累的$L_1$誤差如上圖所示.在這個背景下,為了改善這種情形原文作者提出了`GEAR`用來減少KV cache量化的估計誤差.

## 深入分析GEAR細節


### 前置知识

(i) 基础量化方法

比如说我们有一个tensor $X\in R^{n\times d}$ 作为输入, 想要将这样一个输入的tensor做一个带宽(bandwidth)为b的量化操作,可以描述如下:

$$
Quant_{b,g}^{per-token}=\dfrac{X_{\mathcal{G}_i}-min(X_{\mathcal{G}})}{max(X_{\mathcal{G}})-min(X_{\mathcal{G}})}\times (2^b-1)
$$

其中 $g$ 是指一个量化分组的size; 通常 $g$ 取得越小量化效果越好,但是与此同时g取得越多需要保存的缩放因子也越多会导致内存消耗变大.

(ii) MHA 多头注意力机制

关于多头注意力的分析前面的文章以及分析过不少了,这里仅给出形式化的公式:

$$
\begin{align}
MHA(X)&=concat(H^{(1)},H^{(2)},...,H^{(h)})\cdot W_O\\
H_i&=Softmax(\dfrac{Q^{(i)}K^{(i)T}}{\sqrt{d_H}})\cdot V^{(i)}\\
\end{align}
$$



### GEAR的總體框架

GEAR的整體思路其實很簡單,主要可以描述為以下三步:

(i) 首先對KV cache採用一個常規的量化方法(比如將全進度float16的kv值全部轉儲為int2的類型),但是這必然會導致精度的大幅降低.

(ii) 然后引入一个低秩矩阵来高效的估计量化之后的残差;

(iii) 最后再引入一个稀疏矩阵来补全一些异常值导致的极个别的大额误差;

省流版: 在原来粗暴量化的基础上,整体绝大部分的误差是通过引入一个低秩矩阵来解决的,而一些异常值是通过一个稀疏矩阵来恢复的;

- 符号规定:

量化之后的kv cache矩阵为 $\hat{\mathcal{D}}$; 上文提及的低秩矩阵记作 $\mathcal{L}$; 用于捕捉补偿少部分异常值的稀疏矩阵记作 $\mathcal{S}$;

- 基本策略:

给定一个待处理的tensor $\mathcal{X}\in \{ K_t,V_t\}$ ,我们的策略就是上文提及的三种量化策略之后得到的三部分矩阵, 然后最小化 $\mathcal{X}$ 和上述三部分的距离;所以实际上, 这个任务可以描述为:

$$
minimize\|\mathcal{X}-\hat{\mathcal{D}}-\mathcal{L}-\mathcal{S}\|
$$

(i) 我们都知道过大或者过小的异常值会对量化过程的精度造成极大的影响,所以最佳的策略是在量化之前先进行一次异常值提取, 具体而言:

$$
\begin{align}
\mathcal{S}&=Filter_S(\mathcal{X})\\
Filter_S(\mathcal{X})&=\left\{
\begin{array}{l}
\mathcal{X_{ij}},s.t. \mathcal{X}=K_t and \mathcal{X}_{ij}\text{in top/buttom} \frac{s}{2}\% \text{of the j-th channel}{\mathcal{X}_{*j}} \\
\mathcal{X_{ij}}, s.t. \mathcal{X}=V_t and \mathcal{X}_{ij}\text{in top/buttom} \frac{s}{2}\% \text{of the i-th token} {\mathcal{X}_{i*}}\\
0,  s.t.else.
\end{array}
\right.\\
\end{align}
$$

在异常值提取完成之后,再接着进行量化处理:

$$
\hat{\mathcal{D}}=Quant_{b}^{\text{Selected Scheme}}(\mathcal{X}-\mathcal{S}).
$$

这样的思路其实在之前早已被应用于LLM的权重量化上, 但是相比于对于权重(weight)量化而言, kv cahce拥有更多的
异常值(outliers),使得异常值提取的重要性更大了;


(ii) 提取完成异常值之后再进行低秩矩阵误差估计;


根据上文的说法, 我们定义的低秩残差为 $\mathcal{R}=\mathcal{X}-(\hat{\mathcal{D}+\mathcal{S}})\in\mathcal{R}^{n\times d}$;

然后我们将上述低秩残差分作 $H$ 个多头子矩阵, 其中 $\mathcal{R}_h$ 是第h个头的残差矩阵: $\{\mathcal{R}_h=\mathcal{R}[:,(h-1)d_H:hd_H]\}$


设 $\mathcal{R}_h$ 的奇异值分解形式如 $\sum_{i=1}^{k}\sigma_i\mu_i m_i^T$, 其中 $\sigma_1>\cdots>\sigma_k$ 是 $\mathcal{R}_h$ 的奇异值, $\mu_i$ 和 $m_i$ 为对应的特征向量;


<center>
  <img src="/pics/singluar123.png" width="40%">
</center>






























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


上式中 $E_k=(X-\sum_{i\neq k}d_j\cdot x_j)$ 被定義為殘差;此時最優化問題可以被描述為 $min_{d_k}\|E_k-d_k\cdot z_k\|$ ,這顯然是一個最小二乘問題,可以直接用最小二乘法就可以解決這個問題.

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
  <img src="/pics/sakiko_plane.jpg" width="40%">
</center>




