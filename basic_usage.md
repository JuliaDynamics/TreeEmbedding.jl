# TreeEmbedding.jl 

This package provides a (Monte Carlo Decision Tree Search)[https://link.springer.com/article/10.1007/s11071-022-07280-2] to achieve optimal time-delay embedding of time series. The embedding of a uni- or multivariate time series $s_i$ 

$$\vec{v}(t) = \bigl( s_{i_1}(t-\tau_1), s_{i_2}(t-\tau_2),\ldots,s_{i_m}(t-\tau_m) \bigl)$$

is thus chosen so that the delays $\tau_i$, and time series $i_j$ are optimal with respect to some objective function $\Gamma$, guided by a delay pre-selection statistic $\Lambda_t$. The iterative optimziation process is computed with a decision tree that is only partially computed using a Monte Carlo sampling approach. The tree is sampled $N$ times to minimize $\Gamma$ with certain heuristics applied that guide the samples to more promising leafs of the tree. 

What an optimal embedding is does in general depended on the application that the practioner has in mind. TreeEmbedding.jl allows its users to freely chose the objective function $\Gamma$ and delay pre-selection statistic $\Lambda_t$. 
## The default embedding 

TreeEmbedding.jl provides a default setting that is what can be considered a good "allrounder" embedding. This embedding is a refinement of the (PECUZAL algorithm)[https://iopscience.iop.org/article/10.1088/1367-2630/abe336]. The first sample from the decision tree is identical to PECUZAL algorithm, every further sample tries minimize the objective function, in this case the $\Delta L$ static from (Uzal et al)[^Uzal2011] further. 

As a first example, let us consider a Lorenz63 system. This is meant as a toy example: we generate
data from a Lorenz63 system and then try to reconstruct the full state space from two of the three observables.

First we import the needed modules and generate a trajectory of the Lorenz system (and discard any transient dynamics)

```julia
using DynamicalSystems, TreeEmbedding, Random

# Check Lorenz System
Random.seed!(1234)
ds = Systems.lorenz()
data = trajectory(ds,50, Ttr=100.)[:,1:2]
```

Then, we simply call the default embedding option with $N=50$ trials:

```julia
tree = mcdts_embedding(data, 50)
```

This decision tree search is computionally relatively expensive, you should expect this to take a few minutes. If the keyword 'verbose=true` is used, the current state of the optimization is printed after each sample. After completion it will directly return the best embedding it found: 

```julia
Embedding tree with current best embedding: L=-1.0529700545830651 - full embd. τ=[0, 8, 0] ,i_ts=[1, 2, 2]
```

Thus in the cast we can reconstruct the full system from the first two components with the embedding vector $\vec{v}(t) = \bigl( s_{1}(t), s_{2}(t), s_{2}(t-8) \bigl)$. 

## Configuring the Embedding 

TreeEmbeddding.jl provides much more than just a refinement to the PECUZAL algorithm. It allows to find optimal embedding with different objective functions for a wide variety of applications. Configuring the embedding, breaks down in three parts. 

### Choosing the Objective Function 

The objective function $\Gamma$ quantifies the goodness of a reconstruction, given that
delays $\tau_j$ have been estimated. The embedding process is thought of as an iterative process, starting with an unlagged (given) time series $s_{i_1}$, i.e., $\tau_1 = 0$. In each embedding
cycle $D_d, [d=1,\ldots,m]$ a time series $s_{i_d}$ lagged by $\tau_d$, gets appended to obtain the actual reconstruction vectors $\vec{v}_d(t) \in R^{d+1}$
and these are compared to the reconstruction vectors $\vec{v}_{d-1}(t)$ of the former embedding cycle (if $d=1$, $\vec{v}_{d-1}(t)$ is simply the time series $s_{i_1}$).
This comparison is usually achieved by the amount of false nearest neighbors (FNN), some other
neighborhood-preserving-idea, or more ambitious ideas as the $L$ function from Uzal et. al.

TreeEmbedding.jl predefines four different objective functions: 

* The $L$ function from Uzal et al.[^Uzal2011]: [`L_statistic`](@ref)
* The FNN statistic based on Hegger & Kantz [^Hegger1999]: [`FNN_statistic`](@ref)
* A loss function based on the correlation coefficient of the
convergent cross mapping, from Sugihara et al. [^Sugihara2012]: [`CCM_ρ`](@ref)
* A loss based on a prediction performed with the current reconstruction, see [`Prediction_error`](@ref)

All of these can be directly initialized with their default parameters e.g. by `L_statistic()`, but also further adjusted. For that please see the reference of the individual objective functions. Further objective functions can be defined as subtypes of [`AbstractLoss`](@ref). Most importantly they must have a method [`compute_loss`](@ref) attached. 
### Choosing the Delay Preselection Statistic 

A statistic pre selecting which delays $\tau_j$ to consider for the reconstruction. Usually $\tau_1 = 0$, i.e., the first component of
$\vec{v}(t)$ is the unlagged time series $s_{i_1}$. For embedding a univariate time series,
$s_{i_1}=\ldots=s_{i_m}=s(t)$, the approach to choose $\tau_2$ from the
first minimum of the auto-mutual information is most common. All consecutive delays are then simply integer multiples of $\tau_2$. Other ideas based on different statistics like the auto-correlation
function of the time series have been suggested. However, by setting $\tau_j, j>2$ to multiples of $\tau_2$, one ignores the fact that this measure of independence strictly holds only for the first
two components of reconstruction vectors ($m=2$), even though in practice it works fine for most cases. More
sophisticated ideas, like high-dimensional conditional mutual information and other statistics
some of which include non-uniform delays and the extension to multivariate input data have been presented.

TreeEmbedding.jl predefines two delay preselection statistics:

* The continuity function `⟨ε★⟩` by Pecora et al.[^Pecora2007]: [`Continuity_function`](@ref)
* A given range of delays to consider, without a proper preselection: [`Range_function`](@ref)

### Configuring the Tree Search 

Further, one can modify how the decision tree is sampled and searched. 

As we strive to find a global minimum of the objective function $\Gamma$ and cannot compute the full embedding tree, we proceed by sampling the tree. We randomly sample the full tree, for each embedding cycle we compute the change in the objective function $\Gamma$ and pick for the next embedding cycle preferably those delays that decrease $\Gamma$ further. Each node $N_d$ of the tree encodes one possible embedding cycle and holds the time series used
$[s_{i_1}, \ldots, s_{i_d}]$, the delays used until this node $[\tau_1, \ldots, \tau_{d}]$, i.e., the current path through the tree up to node $N_d$, and a value
of the objective function $\Gamma_d$. We sample the tree $N$-times in total in a two-step procedure:

* _Expand_: Starting from the root, for each embedding cycle $D_d$, possible next steps $(s_{i_j},\tau_j,\Gamma_j)$ are either computed using suitable statistics $\Lambda_{\tau}$ and $\Gamma$ or, if there were already previously computed ones, they are looked up from the tree. We consider the first embedding cycle $D_2$ and use the continuity statistic $\langle \varepsilon^\star \rangle(\tau)$ for $\Lambda_{\tau}$. Then, for each time series $s_{i}$ the corresponding local maxima of all $\langle \varepsilon^\star \rangle(\tau)$ that determines the set of possible delay values $\tau_2$ corresponding to $D_2$). Then, one of the possible $\tau_2$'s is randomly chosen with probabilities computed with a softmax of the corresponding values of $\Gamma_j$. Due to its normalization, the softmax function is able to convert all possible values of $\Gamma_j$ to probabilities with $p_j=\exp(-\beta \Gamma_j)/\sum_k\exp(-\beta \Gamma_k)$. This procedure is repeated until the very last computed embedding cycle $D_{m+1}$. This is, when the objective function $\Gamma_{m+1}$ cannot be further decreased for any of the $\tau_{m+1}$-candidates. 
* _Backpropagation_: After the tree is expanded, the final value $\Gamma_m$ is backpropagated through the taken path of this trial, i.e., to all leafs (previous embedding cycles $d$), that were visited during this expand, updating their $\Gamma_d$ values to that of the final embedding cycle.

We can modify this tree search in TreeEmbedding.jl be setting different values for $\beta$ in the softmax function or by specifiying a completely different function that chooses the next embedding cycle. A large value for $\beta$ means that almost always the strict minimum of all $\Gamma_j$ is chosen, the tree is thus very "narrow", wheareas $\beta=0$ would mean to chose one delay uniformly random and a very "wide" tree search. 

The choose function can be adjusted by supplying the keyword argument `choose_func` to `mcdts_embedding`. The default for that is `(L)->(TreeEmbedding.softmaxL(L,β=2.))`. Additionally we can specifiy a maximum depth for the tree search with the `max_depth` keyword. 
### Putting Everything Together 

We can put these three configuration steps together to get our personal TreeEmbedding by first specifying the [`MCDTSOptimGoal`](@ref) that holds both the objective function and the delay preselection statistic, e.g. via:  

```julia
optimgoal = MCDTSOptimGoal(L_statistic(),Continuity_function())
```

There are also predefined optimziation goals: 

* The default option, the PECUZAL refinement can also be called directly as `PecuzalOptim()`
* FFN statistic + continuity function `FNNOptim()`
* CCM-causality analysis + range function `CCMOptim()`
* A zeroth order predictor and the continuity statistic `PredictOptim()`

Next, we also have to specify the Theiler window (neighbors in time with index `w` close to the point, that are excluded from being true neighbors), e.g. with mutual information minimum method of DelayEmbeddings.jl by

```julia 
w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w = maximum(hcat(w1,w2))
```

Additionally we set what delays are considered, e.g. all delays up to 100 and how many samples we want to compute during the tree search
```julia 
delays = 0:100
N = 50 
```

Then, we can finally compute our embedding: 
```julia 
tree = mcdts_embedding(data, optimgoal, w, delays, N)
```
