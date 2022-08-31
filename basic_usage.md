# TreeEmbedding.jl 

This package provides a [Monte Carlo Decision Tree Search](https://link.springer.com/article/10.1007/s11071-022-07280-2)[^Kraemer2022] to achieve an optimal time-delay embedding of time series. The embedding of a set of $M$ time series $s_i, i=1,\ldots,M$ 

$$\vec{v}(t) = \bigl( s_{i_1}(t-\tau_1), s_{i_2}(t-\tau_2),\ldots,s_{i_m}(t-\tau_m) \bigl)$$

is chosen such that the delays $\tau_{j}$ and time series $i_j$ are optimal with respect to some objective function $\Gamma$, guided by a delay pre-selection statistic 
$\Lambda_{\tau}$. The iterative optimziation process is implemented as a decision tree that is only partially computed using a Monte Carlo sampling approach. The tree is sampled $N$ times to minimize $\Gamma$. Certain heuristics are applied that guide the samples to more promising leafs of the tree with respect to the underlying objective function $\Gamma$. 

What an optimal embedding is does in general depended on the application that the practitoner has in mind. `TreeEmbedding.jl` allows its users to freely chose the objective function $\Gamma$ and delay pre-selection statistic $\Lambda_{\tau}$. 

```@docs
mcdts_embedding
```

## The default embedding 

`TreeEmbedding.jl` provides a default setting that which can be considered a good "allrounder" embedding. This embedding is a refinement of the [PECUZAL algorithm](https://iopscience.iop.org/article/10.1088/1367-2630/abe336)[^Kraemer2021], which uses the $\Delta L$ statistic from Uzal et al.[^Uzal2011] ([`uzal_cost_pecuzal_mcdts`](@ref), [`uzal_cost`](@ref)) as the objective function $\Gamma$ and the *continuity statistic* from Pecora et al.[^pecora2007] ([`pecora`](@ref)) as the delay pre-selection statistic $\Lambda_{\tau}$ in each embedding cycle. The first sample from the decision tree is identical to the PECUZAL algorithm and every further sample tries to minimize the objective function, i.e. to increase the overall $\Delta L$ over all embedding cycles. 

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
Here `L` denotes the chosen objective/loss-function (*L-statistic* in this case), $τ$ yields the found delay values and $i_ts$ its corresponding time series.
Thus, in this example we can reconstruct the full system from the first two coordinates with the embedding vector $\vec{v}(t) = \bigl( s_{1}(t), s_{2}(t), s_{2}(t-8) \bigl)$. 

!!! note "Reproducibility of results"

    Since the decision tree gets randomly sampled the shown results might differ each time you run the code. For reproducible results a random seed must be set, e.g. 
    ```julia
    Random.seed!(1234)
    ```
 

## Configuring the Embedding 

`TreeEmbeddding.jl` provides much more than just a refinement to the PECUZAL algorithm. It allows to find optimal embedding with different objective functions for a wide variety of applications. Configuring the embedding, breaks down in three parts. 

### Choosing the Objective Function 

The objective function $\Gamma$ quantifies the goodness of a reconstruction, given that
delays $\tau_{j}$ have been estimated. The embedding process is thought of as an iterative process, starting with an unlagged (given) time series $s_{i_1}$, i.e., $\tau_1 = 0$. 
In each embedding
cycle $D_d, [d=1,\ldots,m]$ a time series $s_{i_d}$ lagged by $\tau_d$, gets appended to obtain the actual reconstruction vectors $\vec{v}_d(t) \in R^{d+1}$
and these are compared to the reconstruction vectors $\vec{v}_{d-1}(t)$ of the former embedding cycle (if $d=1$, $\vec{v}_{d-1}(t)$ is simply the time series $s_{i_1}$).
This comparison is made through the objective function and, thus, related to the research question. Hence, the embedding gets optimized with respect to the chosen cost function.

So far, `TreeEmbedding.jl` predefines four different objective functions: 

* The $L$ statistic from Uzal et al.[^Uzal2011]: [`LStatistic`](@ref)
* The False Nearest Neighbor (FNN) statistic based on Hegger & Kantz[^Hegger1999]: [`FNNStatistic`](@ref)
* A loss function based on the correlation coefficient of the
convergent cross mapping, from Sugihara et al.[^Sugihara2012]: [`CCM_ρ`](@ref)
* A loss based on a prediction performed with the current reconstruction, see [`PredictionError`](@ref)

```@docs
LStatistic
FNNStatistic
CCM_ρ
PredictionError
```

All of these can be directly initialized with their default parameters e.g. by `LStatistic()`, but also further adjusted. For that please see the reference of the individual objective functions. Further objective functions can be defined as subtypes of [`AbstractLoss`](@ref). Most importantly they must have a method [`compute_loss`](@ref) attached. 

```@docs
AbstractLoss
compute_loss
```



### Choosing the Delay Preselection Statistic 

A statistic pre selecting which delays $\tau_j$ to consider for the reconstruction can be defined. Usually $\tau_1 = 0$, i.e., the first component of
$\vec{v}(t)$ is the unlagged time series $s_{i_1}$. For embedding a univariate time series,
$s_{i_1}=\ldots=s_{i_m}=s(t)$, the approach to choose $\tau_2$ from the
first minimum of the auto-mutual information is most common. All consecutive delays are then simply integer multiples of $\tau_2$. Other ideas based on different statistics like the auto-correlation
function of the time series have been suggested. However, by setting $\tau_j, j>2$ to multiples of $\tau_2$, one ignores the fact that this measure of independence strictly holds only for the first
two components of reconstruction vectors ($m=2$), even though in practice it works fine for most cases. More
sophisticated ideas, like high-dimensional conditional mutual information and other statistics,
some of which include non-uniform delays and the extension to multivariate input data, have been presented.

So far `TreeEmbedding.jl` predefines two delay preselection statistics:

* The continuity function `⟨ε★⟩` by Pecora et al.[^Pecora2007]: [`Continuity_function`](@ref), [`pecora`](@ref)
* A given range of delays to consider, without a proper preselection: [`Range_function`](@ref)

```@docs
Continuity_function
Range_function
```

### Configuring the Tree Search 

Further, one can modify how the decision tree is sampled and searched. 

As we strive to find a global minimum of the objective function $\Gamma$ and cannot compute the full embedding tree, we proceed by sampling the tree. We randomly sample the full tree, for each embedding cycle we compute the change in the objective function $\Gamma$ and pick for the next embedding cycle preferably those delays that decrease $\Gamma$ further. Each node $N_d$ of the tree encodes one possible embedding cycle and holds the time series used
$[s_{i_1}, \ldots, s_{i_d}]$, the delays used until this node $[\tau_1, \ldots, \tau_{d}]$, i.e., the current path through the tree up to node $N_d$, and a value
of the objective function $\Gamma_d$. We sample the tree $N$-times in total in a two-step procedure:

* _Expand_: Starting from the root, for each embedding cycle $D_d$, possible next steps $(s_{i_j},\tau_j,\Gamma_j)$ are either computed using suitable statistics $\Lambda_{\tau}$ and $\Gamma$ or, if there were already previously computed ones, they are looked up from the tree. We consider the first embedding cycle $D_2$ and use the continuity statistic $\langle \varepsilon^\star \rangle(\tau)$ for $\Lambda_{\tau}$. Then, for each time series $s_{i}$ the corresponding local maxima of all $\langle \varepsilon^\star \rangle(\tau)$ that determines the set of possible delay values $\tau_2$ corresponding to $D_2$). Then, one of the possible $\tau_2$'s is randomly chosen with probabilities computed with a softmax of the corresponding values of $\Gamma_j$. Due to its normalization, the softmax function is able to convert all possible values of $\Gamma_j$ to probabilities with $p_j=\exp(-\beta \Gamma_j)/\sum_k\exp(-\beta \Gamma_k)$. This procedure is repeated until the very last computed embedding cycle $D_{m+1}$. This is, when the objective function $\Gamma_{m+1}$ cannot be further decreased for any of the $\tau_{m+1}$-candidates. 
* _Backpropagation_: After the tree is expanded, the final value $\Gamma_m$ is backpropagated through the taken path of this trial, i.e., to all leafs (previous embedding cycles $d$), that were visited during this expand, updating their $\Gamma_d$ values to that of the final embedding cycle.

We can modify this tree search in `TreeEmbedding.jl` be setting different values for $\beta$ in the softmax function or by specifiying a completely different function that chooses the next embedding cycle. A large value for $\beta$ means that almost always the strict minimum of all $\Gamma_j$ is chosen, the tree is thus very "narrow", wheareas $\beta=0$ would mean to choose one delay uniformly random and a very "wide" tree search. 

The choose function can be adjusted by supplying the keyword argument `choose_func` to `mcdts_embedding`. The default for that is `(L)->(TreeEmbedding.softmaxL(L,β=2.))`. Additionally, we can specifiy a maximum depth for the tree search with the `max_depth` keyword. 

### Putting Everything Together 

We can put these three configuration steps together to get our personal TreeEmbedding by first specifying the [`MCDTSOptimGoal`](@ref) that holds both the objective function and the delay preselection statistic. 

```@docs
MCDTSOptimGoal
```

This is the key constructor for using [`mcdts_embedding`](@ref). For example, when we want to use the PECUZAL idea, we combine the $L$-statistic (objective/loss-function $\Gamma$) 
and the *continuity statistic* (delay pre-selection statistic $\Lambda_{\tau}$) via:

```julia
optimgoal = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.LStatistic(),TreeEmbedding.Continuity_function())
```

There are some predefined optimziation goals: 

* The above mentioned PECUZAL refinement can also be called directly as `optimgoal = PecuzalOptim()`
* FFN statistic + continuity function `optimgoal = FNNOptim()`
* CCM-causality analysis + range function `optimgoal = CCMOptim()`
* A zeroth order predictor and the continuity statistic `optimgoal = PredictOptim()`

Next, we also have to specify the Theiler window (neighbors in time with index `w` close to the point, that are excluded from being true neighbors), e.g. with mutual information minimum method of `DelayEmbeddings.jl` by

```julia
w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w = maximum(hcat(w1,w2))
```

Additionally, we set what delays are considered, e.g. all delays up to 100 and how many samples we want to compute during the tree search
```julia
delays = 0:100
N = 50 
```

Then, we can finally compute our embedding: 
```julia
tree = mcdts_embedding(data, optimgoal, w, delays, N)
```

## References

[^Kraemer2021]: Kraemer, K.H., Datseris, G., Kurths, J., Kiss, I.Z., Ocampo-Espindola, Marwan, N. (2021). [A unified and automated approach to attractor reconstruction. New Journal of Physics 23(3), 033017](https://iopscience.iop.org/article/10.1088/1367-2630/abe336).

[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).

[^Pecora2007]: Pecora, L. M., Moniz, L., Nichols, J., & Carroll, T. L. (2007). [A unified approach to attractor reconstruction. Chaos 17(1)](https://doi.org/10.1063/1.2430294).

[^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).

[^Sugihara2012]: Sugihara et al., [Detecting Causality in Complex Ecosystems. Science 338, 6106, 496-500](https://doi.org/10.1126/science.1227079).

[^Kraemer2022]: Kraemer, K.H., Gelbrecht, M., Pavithran, I., Sujith, R. I. and Marwan, N. (2022). [Optimal state space reconstruction via Monte Carlo decision tree search. Nonlinear Dynamics](https://doi.org/10.1007/s11071-022-07280-2).

