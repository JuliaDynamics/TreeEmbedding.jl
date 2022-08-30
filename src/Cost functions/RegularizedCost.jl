"""
    RegularizedCost{T,S} <: AbstractLoss

Consists of another `AbstractLoss` and a `regularization` which is a function `(L_old, depth) -> L_new`` 

"""
struct RegularizedCost{T,S} <: AbstractLoss
    cost_function::T
    regularization::S
end 
 
init_embedding_params(Γ::RegularizedCost, N::Int) = init_embedding_params(Γ.cost_function, N)

Base.push!(children::Array{Node,1}, n::EmbeddingPars, Γ::RegularizedCost, current_node::AbstractTreeElement) = push!(children, n, Γ.cost_function, current_node)

threshold(L::RegularizedCost) = threshold(L.cost_function) 

get_embedding_params_according_to_loss(Γ::RegularizedCost, embedding_pars::Vector{EmbeddingPars}, L_old) = get_embedding_params_according_to_loss(Γ.cost_function, embedding_pars, L_old)

function compute_loss(Γ::RegularizedCost, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; L_old, metric=Euclidean(), kwargs...) where {P, D, T}

    L, max_idx, temp = compute_loss(Γ.cost_function, Λ, dps, Y_act, Ys, τs, w, ts, τ_vals, ts_vals; metric=metric, kwargs...)

    depth = length(τ_vals)
    regularize_L!(L, L_old, depth, Γ.regularization)

    return L, max_idx, temp 
end 

function regularize_L!(L, L_old, depth, func)
    for i in eachindex(L)
        L[i] = func(L[i], L_old, depth)
    end 
end 