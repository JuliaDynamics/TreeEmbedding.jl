# False-Nearest Neigbour based Embedding for Recurrence Analysis 

In the following, we will demonstrate how to perform an embedding of a 8-dimensional Lorenz 96 system (L96) for a recurrence analysis. We will determine recurrence quantification analysis (RQA) measures from the true, fully known system and then compare them to the RQA measures computed from an embedding that can only use the 2nd, 5th and 7th dimension of the L96. [^Kraemer2022] showed that a False-nearest neighbour approach works best in this case, so we will use TreeEmbedding with an FNN cost function and the continuity statistic to pre-select delay. For a comprehensive and more systematic evaluation of the different embedding algorithms for RQA, see [^Kraemer2022].

First, we import all packages, generate the data from L96 and compute the RQA for the fully known system. For that purpose, we also estimate a Theiler window using the minimum in mutual information. The recurrence analysis is performed with the `RecurrenceAnalysis` package.

```julia 
using TreeEmbedding
using DynamicalSystems
using DelayEmbeddings
using RecurrenceAnalysis
using Statistics

begin # generate data 
    N = 8 # number of oscillators
    dt = 0.1 # sampling time
    total = 5000  # time series length
    u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298] # initial conditions 
    lo96 = Systems.lorenz96(N, u0; F = 3.8)
    data = trajectory(lo96, total*dt;  Δt = dt, Ttr = 2500*dt)
end 

begin # reference RQA 
    ε = 0.05  # recurrence threshold
    lmin = 2   # minimum line length for RQA

    # theiler window estimation
    w1 = DelayEmbeddings.estimate_delay(data[:,2], "mi_min") # this is the time series we take as an example in the univariate case 
    theiler = w1 

    R_ref = RecurrenceMatrix(data, ε; fixedrate = true)
    RQA_ref = rqa(R_ref; theiler = theiler, lmin = lmin)
end 
```

Now, a benchmark for our improved TreeEmbedding we also compute an RQA from a traditional time delay embedding using the FNN approach.

```julia
dmax = 10   # maximum dimension for traditional tde
    
data_sample = data[:,2] # has to be univariate 
𝒟, τ_tde2, E = optimal_traditional_de(data_sample, "fnn"; dmax = dmax)

R_tde = RecurrenceMatrix(𝒟, ε; fixedrate = true)
RQA_tde = rqa(R_tde; theiler = theiler, lmin = lmin)
RQA_tde = hcat(RQA_tde...)
```

Finally, we compute our TreeEmbedding. As an optimziation goal we set it up to minimize the [`FNNStatistic`](@ref) and we use the [`ContinuityFunction`](@ref) to pre-select possible delay values. If you want to get some quick results, reduce `trials` e.g. to `5`. For `trials=50` as given here, the embedding will need some time to compute. 

```julia
# TreeEmbedding RQA
trials = 50 # trials for MCDTS
taus = 0:100 # possible delays
    
# pick one time series
t_idx = [2,4,7]
data_sample = data[:,t_idx]

optim = TreeEmbedding.FNNOptim()
tree = mcdts_embedding(Dataset(data_sample), optim, theiler, taus, trials; max_depth = 20)
best_node = TreeEmbedding.best_embedding(tree)

𝒟_tree = genembed(data_sample, best_node.τs, best_node.ts)
R_tree = RecurrenceMatrix(𝒟_tree, ε; fixedrate = true)
RQA_tree = rqa(R_tree; theiler = theiler, lmin = lmin)
RQA_tree = hcat(RQA_tree...)
```

Let's compare the outputs! How well does the TreeEmbedding compared to the traditional TDE? 

| RQA metric | ground truth | TreeEmbedding  | traditional TDE |
|------------|--------------|----------------|-----------------|
|     DIV    | 0.00108342   | **0.00104167** | 0.00125786      |
|     LAM    | 0.992468     | **0.993599**   | 0.983415        |
|    Vmax    | 6.0          | **6.0**        | 7.0             |
|    VENTR   | 1.06367      | **1.28383**    | 1.48477         |
|    Lmax    | 923.0        | **960.0**      | 795.0           |
|     MRT    | 68.0759      | **68.199**     | 65.9199         |
|    NMPRT   | 135143.0     | **139733.0**   | 125830.0        |
|     RR     | 0.049627     | **0.0496247**  | **0.0496288**   |
|     RTE    | 1.52361      | **1.506879**   | 1.8579053       |
|     TT     | 3.45256      | **3.45444**    | 3.4193568       |
|      L     | 21.9827      | **22.599360**  | 9.591337        |
|    ENTR    | 3.26345      | **3.42507916** | 2.254923        |
|     DET    | 0.991549     | **0.99162374** | 0.957285        |
|    TREND   | 0.000221804  | **-0.0001053** | 0.0007181       |

We've seen how the RQA metrics are much closer to the ground truth when an embedding with TreeEmbedding is performed than with a traditional TDE, some fit the ground truth almost exactly. For a more comprehensive application and comparision between various different embedding algorithms and for a wide parameter range of the Lorenz 96 system, please see [^Kraemer2022].
# References 

[^Kraemer2022]: Kraemer, K.H., Gelbrecht, M., Pavithran, I., Sujith, R. I. and Marwan, N. (2022). [Optimal state space reconstruction via Monte Carlo decision tree search. Nonlinear Dynamics](https://doi.org/10.1007/s11071-022-07280-2).

