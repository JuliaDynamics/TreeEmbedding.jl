# Regularization of the Embedding 

For some applications it might be desirable to get embeddings that do not only strictly minimize the cost function but also take in to account to yield a well working embedding with less embedding dimensions. This is espacially true when using it for prediction tasks, as this helps to avoid overfitting. To accomplish this, it is possible to add a regularization to any cost function. 

The cost function $\Gamma$ is then regularized in the following way

$$ \Gamma_{reg} = $\Gamma_{old} * (1 \pm d\cdot\lambda, $$

where $d$ is the embedding dimension and $\lambda << 1$ is the regularization constant. The plus sign applies to cost functions that yield positive numbers (the usual case), and the minus sign to cost functions that yield negative numbers (like the [`LStatistic`](@ref)). It thus penalizes embeddings with many embedding dimensions and favours those embeddings with fewer embedding dimensions. Typical values for $\lambda$ are ...

TreeEmbeeding.jl provides [`RegularizedCost`](@ref) for this purpose. To use it, it must be initialized with the cost function that should be regularized like this: 

```julia 
original_cost = LStatistic() # could also be any other cost function <: AbstractLoss 
lambda = 1e-2 # regularization constant 
regularized_cost = RegularizedCost(original_cost, lambda)
```

In the following, we demonstrate this for a prediction task

```julia 
# code here 
```