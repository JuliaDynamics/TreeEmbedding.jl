"""
    RegularizedCost{T,S} <: AbstractLoss

Consists of another `AbstractLoss` and a `regularization` which is a function `(L_old, depth) -> L_new`` 

"""
struct RegularizedCost{T,S} <: AbstractLoss
    cost_function::T
    regularization::S
end 

init_embedding_params(Γ::RegularizedCost, N::Int) = init_embedding_params(Γ.cost_function, N)

function compute_loss(Γ::RegularizedCost, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    L, max_idx, temp = compute_loss(Γ.cost_function, Λ, dps, Y_act, Ys, τs, w, ts, τ_vals; metric=metric, kwargs...)
    
    depth = length(τ_vals) # Hauke: is this correct, is this the depth of the tree at that point?
    regularize_L!(L, depth, Γ.regularization)

    return L, max_idx, temp 
end 

function regularize_L!(L, depth, func)
    for i in eachindex(L)
        L[i] = func(L[i], depth)
    end 
end 
