# Neural networks

### [LSTM explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

This is very informative, but the key idea is just to remember the general
structure of a LSTM:
$$C_t = C_{t-1} \odot \sigma(T_1 [h_{t-1}; x_t]) + \sigma(T_2 [h_{t-1}; x_t])
\odot \tanh(T_3[h_{t-1}; x_t])$$
$$h_{t} = \sigma(T_4 [h_{t-1}; x_t]) \odot \tanh(C_t)$$
in which $C_t$ is the long term memory, whereas $h_t$ is the output and
short-term memory and $T_i$ are affine transformations. 

### [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427v1)

Pretty similar idea to Neural Message Passing for Quantum Chemistry, but less
general that that.

### [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)

The idea is that in case of over-parametrized model, the weights don't really
change much (mainly by curse of dimensionality and the fact that little
variations in the weights corresponds to large effects like a snowball effect). Hence, the neural network through training could be considered as a linear approximation around the initialization. In [On the (non-) robustness of two-layer neural networks in different learning regimes](https://arxiv.org/abs/2203.11864), they also provide that an initialization whose output is $0$ is necessary to have a robust model in this ''lazy-training'' regime.

### [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212v2)

The central idea is the development of a general framework in the context of
RNN, that expands on the ideas of [A simple neural network module for relational
reasoning](https://arxiv.org/abs/1706.01427v1) and other papers. The idea is
explained at page 3, they describe a network in which the functionals $M_t$ and
$U_t$ are trainable and describe on a graph $G$ for $t \in \{0, \dots, T\}$
time-steps the following algorithm: $$m^{t+1}_v = \sum_{w \in N(v)} M_t(h^t_v,
h^t_w, e_{vw})$$ $$h^{t+1}_v = U_t(h^t_v, m^{t+1}_v)$$
In this case $N(v)$ represent the neighbors of each vertex $v$ and $e_{vw}$
describes a certain edge feature between the nodes $v, w$. The output of the
network is then read through a function $g$ (which could be parametrised), 
$$\hat{y} = g(\{h_v^T \mid v \in G\})$$. 

Simple remarks and little examples may be required to fully capture the idea:
suppose the data is composed by words in a sentence, like "It rains, therefore I
open my umbrella".
The graph might have an edge that connects "rains" with "umbrella" or "open",
similarly "open" with "I" respectively motivated by a logical connection and a
grammatical one. Note that because the parameters of the neural networks are needed to 
describe the functions $M_t$ and $U_t$, we are not really worried about the
dimension of the graph in input! So, we simulate this message passing functions,
for example by using a non-linear function applied to an affine transformation
$M_t = \tanh(W_t [h^t_v; h^t_w] + b_t)$, and similarly for $U_t$. Note that $g$
might depend on the structure of the graph, but we could also just consider a
projection to the $n$-th coordinate or the average of the outputs to overcome
the graph structure requirement.

<!-- TODO if this is correct, I might change it later -->
---
**NOTE**

Note that a RNN is just a MPRNN with a directed graph relation 
$G = \{ v \in \{ 1,..., N\} \}$ and edges $v \to w$ if $|v - w| = 1$, and
$e_{vw} = 1$ if and only if $w=v-1$, otherwise it is $0$ (that is we allow
previous knowledge, but we do not acknowledge the future). The function
$M_t(h^t_v, h^t_w, e_{vw}) = e_{vw} \sigma(A[h^t_v, h^t_w])$ for some trainable
affine transformation $A$ and the time-steps $T = N$, then the output is $g(G) =
h_{N}^N$ is the output of the RNN. Note that in this manner, we compute the same
outputs $h_k^t$ $(N-k)$-times, which is kinda wasteful.

---

### [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473v7)

TODO: read it more carefully
They propose a bidirectional RNN architecture... which idk why it should be
meaningful: they analyze a sequence from the start to the end and
simultaneously from the end to the beginning. This kinds of remind me the
Feynman-Kac formula, basically we can simulate backwards a PDE using an SDE.

### [RECURRENT NEURAL NETWORK REGULARIZATION](https://arxiv.org/abs/1409.2329v5)

The idea is straightforward: apply dropout inside LSTM architectures.
However, we do not want to corrupt the ''memory'' or
long term relations in the data (or middle-steps of the network). For this
reason the dropout is applied to the LSTM module's output.
Pretty simple application, but with highly beneficial results. The same
principle of not ''corrupting'' the long-term relations, could be applied to
residual steps or short-cut connections: do not apply dropouts to the ''skip''
but only to the ''message modificator'' network.
To be more precise, the functional $F$ may use dropouts to enhance stability, 
$$h_{t+1} = G(h_{t}) + F(h_t)$$
but $G$ should be just a simple passing information on $h_t$, as it was
described in [Identity Mappings in Deep Residual
Networks](https://arxiv.org/abs/1603.05027), $G(h_t) \approx h_t$ for optimal
training results.


### [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361v1)

The scaling laws are related to the context of transformer and to language
modeling. The key points are explained in the summary, I just selected the most 
interesting:          
1. Performance depends strongly on model parameters, dataset size and compute
   used in training. Depth and width does not seem to change results
significantly.           
2. When increasing by $8$-times the number of parameters, the data set must
   increase by roughly $5$-times to avoid penalties in training.        
3. Larger models are more sample-efficient: they require less samples to achieve
   better results. Moreover, larger models better learn distributions by not
achieving convergence on the target dataset.      

Note to (1): it could be caused by the fact that the networks used
are too large to begin with, hence the number of parameters is more than enough
to suffice the lack of ''structural'' differences.       

Note to (3): the fact that very large models seem ''better'' when
they are not close to convergence on the training dataset could be because
convergence is not really optimal. See the difference between a student that
learn by rote-memorization and a student that learns by grokking the material.

### [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239v2)

Definitely worth checking out [https://en.wikipedia.org/wiki/Diffusion_model](Wikipedia page) to better understand the paper. 

The idea is the following:
1. The initial value $x_0 \sim q$ for some distribution $q$.
2. Add iteratively some fixed known noise (gaussian) $x_{t+1} = \sqrt{1-\beta_t} x_t + \sqrt{\beta_t} \varepsilon_t$, with $\varepsilon_t \sim N(0, I)$, therefore $x_{t} \mid x_0  \sim N(\sqrt{a_t} x_0, (1-a_t)I)$ for some parameter $a_t$ and $x_{t-1} \mid x_t, x_0 \mid N(\mu(x_t, x_0), bI)$ for some value $\mu(x_t,x_0)$ and $b$.
3. The point is that we train a neural network to estimate $\mu(x_t, x_0)$
   (because the data $x_t,x_0$ are known and try to learn the mean parameter and
the variance parameter for each time $t$. 
That's it, that's the magic of the diffusion model. How to learn the model that finds the optimal parameters $\mu_t$ is quite straightforward: start from the KL divergence between the distribution $p_{\theta,t}$ (the distribution parametrized by the neural network that should denoise the image) and optimize it against the distribution $q_{t\mid x_0}$ which adds the noise to the initial image. Because $q_{t\mid x_0} \sim N(0, I)$ when $t \to \infty$ and fixed $x_0$ is basically a normal distribution, we obtain a quite simple loss function (up to some scaling constant and translations), 
$$ L_t = \mathbb{E}_{x_0 \sim q, z\sim N(0,I)}[\| \varepsilon_{\theta}(x_t,t) -
z\|^2], \quad \forall t \le T,$$
with $\varepsilon_{\theta}(x_t,t)$ being a sample from the distribution
$N(\mu_\theta(x_t, t), \sigma^2_{t,\theta} I)$

### [An Evolved Universal Transformer Memory](https://arxiv.org/abs/2410.13166)

The idea is straightforward: they train a binary classifier on the KV-cache to 
remove pointless tokens and reduce memory consumption in the KV-cache. 

--- 

**Note**
this is basically a memory-pruning rather than a memory augmentation (see 
[HMT: Hierarchical Memory Transformer for Long Context Language Processing](https://arxiv.org/abs/2405.06067) 
for a memory augmentation procedure that increase the context-length). 

---

The methodology is quite more involved and there are quite a few interesting
details: Short-Time Fourier Transform (STFT), Backward Attention Memory models
(BAM).

They state the attention mechanism as 

\[
\text{attention}_M(Q, K, V) = AV, \quad A= \textrm{softmax}\left( M
\frac{QK^\top}{\sqrt{d}}\right).
\]

They define the Backward Attention Memory models (BAM) which is basically the
attention mechanism with a 'counter-causal' mask $\tilde{M}$, i.e. $M_{ij} = 1$
iff $i < j$, 

\[ \textrm{BAM}(\omega) =
\textrm{linear}(\textrm{attention}_{\tilde{M}}(K_{\Omega}, V_{\Omega},
Q_{\Omega})).\]

The goal is that BAM wants to know the influence of the previous
tokens on the next, we don't want to predict the next token given the next. The
BAM model are used on the space of $\omega_i$ which are feature vectors computed 
as an exponential moving average of the Short-Time Fourier Transform of each
token.
Then, if the classifier given by the BAM model returns a negative value, the
KV-cache relative to the token is discarded, otherwise it is preserved.

### [The Internal State of an LLM Knows When It’s Lying](https://arxiv.org/abs/2304.13734)

The idea is to use a binary classifier trained on the internal state of the LLM
(likely central or almost ending embedding like between 75% and 50% from the
beginning of the LLM attention stack), on a dataset of true-false sentences.
Then, the binary classifier should be able to generalize enough that in
general is able to understand when the model is hallucinating.  

The core of the paper is the description of the true-false dataset. The usage
of the binary classifier is justified with respect to the given dataset.

---

**Note**

They provide very weak empirical results, the example are always formulated
in a similar manner to that of the dataset.

---

### [No More Adam: Learning Rate Scaling at Initialization is All You Need](https://arxiv.org/abs/2412.11768)

I tried with a toy model in classification setting, i.e. feedforward with relu
activation and a simple dropout to reduce overfitting, ADAM used 2 epochs on a
40 elements dataset, whereas this optimization algorithm didn't find any
optimization path: the loss exploded in training. The issue is that the gSNR
computed with the code provided was not ''stable'', it was high ($10^4$) for
the first layer and low for other layers ($100$). Numerically it is unstable,
$\varepsilon$ is highly influential in the training procedure. I conclude that
the choice of the optimization hyper-parameters are quite influential and needs
lots of tweaking or the code provided is bogus or the results are fake. The
result seems consistent with the discussion on [HN](https://news.ycombinator.com/item?id=42448193).

### [Titans: Learning to Memorize at Test Time](https://doi.org/10.48550/arXiv.2501.00663)

There are 2 main contributions: 
1. The addition of an in context short-term memory to the attention layer,
   namely they use a small MLP to encode complex relationships, this is updated
   at inference by decreasing the "surprise" loss (i.e. prediction of the next
   token memory vs the observed next token), while it is propagated its inference
   after observation.
2. a "prior" token that is a sequence of learnable, task dependent parameters
   tokens that are added at the beginning of an input

--- 

**Note**
The first contribution is rather simple but effective. It seems to be loosely
related to the ideas of the Free Energy Principle in computational psychology,
namely that the brain is a prediction machine that tries to align internal
representation and observed representation by minimizing both the complexity of
the explanation and the 'dis'-alignment of the prediction.

---


---

**Note**
The ideas seem an extension upon [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620).
The main motif is to use the backpropagation as a way to encode memory into a simple system, 
i.e. fit the 'right' parameters on the learning time axis of the parameters.

---


### [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/abs/2501.09732)

The idea is trivial: the diffusion models start from an initial noise, what if
we tweak the initial point in a way such that it is the 'optimal' starting point 
for the conditional generative task? The answer is that the initial point is
rather important as well as the number of de-noising steps. The point of this
paper is just empirical to show the scaling laws of commonly used models.

--- 

**Note** 
It does not seem that they point out some kind of 'distribution' of the optimal 
searched points. This could be an interesting application. Namely, if they were
to find a distribution of initial point that is (with high probability) always 
more appealing than the random normal distribution, that could be huge in
reducing the computation! They point to other works related to this question,
but it seems still a 'direct' search / optimization approach rather than a 
statistical one.

---

