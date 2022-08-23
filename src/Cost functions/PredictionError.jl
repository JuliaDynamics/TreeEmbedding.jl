# All functionality corresponding to Predictionerror as Loss function Γ.

## Constructors:

abstract type AbstractMCDTSpredictionType end

abstract type AbstractPredictionLoss{P} end

abstract type AbstractPredictionMethod end

abstract type AbstractLocalPredictionMethod{T} <: AbstractPredictionMethod end

"""
    PredictionError <: AbstractLoss

Constructor for the PredictionError loss function.

## Fieldnames
* `PredictionType::MCDTSpredictionType`: Determines the prediction type by
    setting a prediction-method and the way the prediction error is measured,
    see [`MCDTSpredictionType`](@ref).
* `threshold::Float`: A threshold for the sufficient minimum prediction error
    for the current embedding. When the prediction error, specified in
    `PredictionType`, falls below this threshold in an embedding cycle the
    embedding stops.
* `samplesize::Real = 1.`: the fraction of all phase space points
    to be considered in the computation of the prediction error under the given
    `PredictionType`.
* `error_weights::Vector{Real} = [1 1]`: The weights for determining the prediction 
    error. The first element of this Vector is the weight for the insample prediction
    error and the second elements corresponds to the weight for the out-of-sample
    prediction error. By default only the out-of-sample error will be used (i.e. equal 
    weights [1 1]). Since the out-of-sample prediction error is generally larger than 
    the insample prediction error it may make sense to tune these wheight-parameters.
    For specifying the prediction horizon see [`LocalModel`](@ref)  

## Defaults
* When calling `PredictionError()`, a PredictionError-object is created,
    which uses the threshold 0, i.e. no threshold, a zeroth-order predictor
    (see [`MCDTSpredictionType`](@ref), [`PredictionLoss`](@ref) &
    [`LocalModel`](@ref)) and a full phase space sample (samplesize=1).
"""
struct PredictionError <: AbstractLoss
    PredictionType::AbstractMCDTSpredictionType
    threshold::AbstractFloat
    samplesize::Real
    error_weights::Vector{Real}
    # Constraints and Defaults
    PredictionError(x ,y=0, z=1, zz=[1.;1.]) = begin
        @assert y ≥ 0 "A positive threshold for the prediciton loss must be chosen."
        @assert 0 < z ≤ 1. "The samplesize must be in the interval (0 1]."
        @assert length(zz) == 2 && 0 <= zz[1] ≤ 1 && 0 <= zz[2] ≤ 1 "The errorweights 
        must be passed in a vector of length 2. The elements of this vector need to be 
        in the interval [0 1]."
        new(x, y, z, zz)
    end

end
PredictionError() = PredictionError(MCDTSpredictionType())

"""
    MCDTSpredictionType <: AbstractMCDTSpredictionType

Constructor, which determines the way how predictions are made technically.

## Fieldnames
* `loss::AbstractPredictionLoss`: Indicates the way of computing the prediction error.
    See [`PredictionLoss`](@ref) for information on how to construct this object.
* `method::AbstractPredictionMethod`: The method based on the state space reconstruction,
    which makes the actual prediction. See [`LocalModel`](@ref).

## Default settings
* When calling `MCDTSpredictionType()` a MCDTSpredictionType-object is constructed
    with a `local_zeroth`-predictor [`LocalModel`](@ref), using 2 nearest neighbors
    and a 1-step-ahead-prediction. The loss-function is the root mean squared prediction
    error over all components [`PredictionLoss`](@ref).
"""
struct MCDTSpredictionType <: AbstractMCDTSpredictionType
    loss::AbstractPredictionLoss
    method::AbstractPredictionMethod
    # Constraints and Defaults
    MCDTSpredictionType(x,y) = new(x,y)
    MCDTSpredictionType(x) = new(x,LocalModel())
    MCDTSpredictionType() = new(PredictionLoss(), LocalModel())
end


"""
    PredictionLoss <: AbstractPredictionLoss

Constructor, which indicates the way of computing the prediction error. This
object is used for the constructor, which determines the way how predictions are
made methodologically [`MCDTSpredictionType`](@ref).

## Fieldnames
* `type::Int` is an integer, which encodes the type of prediction error:

- For `type = 1` the root mean squared prediction error over the first component,
    i.e. the timeseries, which needs to be predicted, is used. (default)
- For `type = 2` the root mean squared prediction error over all components
    (dimensionality of the state space) is used.
- For `type = 3` the mean Kullback-Leibler Distance of the predicted and the true
    values of the first component, i.e. the timeseries, which needs to be predicted, is used.
- For `type = 4` the mean Kullback-Leibler Distance of the predicted and the true
    values over all components (dimensionality of the state space) is used.

## Defaults
* When calling `PredictionLoss()` a PredictionLoss-object is constructed with
    fieldname `type = 1` (≡root mean squared prediction error over all components)
"""
struct PredictionLoss{t} <: AbstractPredictionLoss{t}
    type::Int
    # Constraints and Defaults
    PredictionLoss(x=1) = begin
        @assert x == 1 || x == 2 || x == 3 || x == 4
        new{x}(x)
    end
end


"""
    LocalModel <: AbstractLocalPredictionMethod

Constructor, which indicates the local state space prediction model.

## Fieldnames
* `method::String`: Could be `"zeroth"` (averaged `Tw`-step-ahead image of the
    `KNN`-nearest neighbors) or `"linear"` (local linear regression on the
    `KNN`-nearest neighbors).
* `KNN::Int`: The number of considered nearest neighbors.
* `Tw_out::Int` : The prediction horizon in sampling units for the out-of-sample prediction.
* `Tw_in::Int` : The prediction horizon in sampling units for the in-sample prediction.
* `trials::Int` : The number of different out-of-sample prediction trials.

## Defaults
* When calling `LocalModel()` a LocalModel-object is constructed with a zeroth
    order prediction scheme, 2 nearest neighbors, a 1-step-ahead in-sample prediction, 
    a 10-step-ahead out-of-sample prediction and 20 out-of-sample trials.
* When calling `LocalModel(method)` a LocalModel-object is constructed with a
    `method`-prediction scheme, 2 nearest neighbors, a 1-step-ahead in-sample prediction, 
    a 10-step-ahead out-of-sample prediction and 20 out-of-sample trials.
* When calling `LocalModel(method,KNN)` a LocalModel-object is constructed with a
    `method`-prediction scheme, `KNN` nearest neighbors, a 1-step-ahead in-sample prediction, 
    a 10-step-ahead out-of-sample prediction and 20 out-of-sample trials.
"""
struct LocalModel{m} <: AbstractLocalPredictionMethod{m}
    method::String
    KNN::Int
    Tw_out::Int
    Tw_in::Int
    trials::Int
    
    # Constraints and Defaults
    LocalModel(x="zeroth", y=2, z=10, zz=1, zzz=20) = begin
        @assert x in ["zeroth", "linear"]
        @assert y > 0
        @assert z > 0
        @assert zz > 0
        @assert zzz > 0
        m = Symbol(x)
        new{m}(x,y,z,zz,zzz)
    end
end




## Functions:

"""
    embedding_trajectory(Γ::AbstractLoss, Y, τ_vals, ts_vals)

Compute actual embedding trajectory with `DelayEmbeddings.genembed`, respects the causailty for the prediction
"""
embedding_trajectory(Γ::PredictionError, Ys, τ_vals, ts_vals) = genembed(Ys, τ_vals .* (-1), ts_vals)

"""
    Return the loss based on a `Tw`-step-ahead local-prediction.
"""
function compute_loss(Γ::PredictionError, Λ::AbstractDelayPreselection, dps::Vector{P}, Y_act::Dataset{D, T}, Ys, τs, w::Int, ts::Int, τ_vals, ts_vals; metric=Euclidean(), kwargs...) where {P, D, T}

    PredictionLoss = Γ.PredictionType.loss
    PredictionMethod = Γ.PredictionType.method
    samplesize = Γ.samplesize
    error_weights = Γ.error_weights

    max_idx = get_max_idx(Λ, dps, τ_vals, ts_vals, ts) # get the candidate delays
    isempty(max_idx) && return Float64[], Int64[], []

    costs_insample = zeros(Float64, length(max_idx))
    costs_out_of_sample = zeros(Float64, length(max_idx))
    for (i,τ_idx) in enumerate(max_idx)
        # create candidate phase space vector for this peak/τ-value
        tau_trials = (τ_vals...,τs[τ_idx-1],)
        ts_trials = (ts_vals...,ts,)
        Y_trial = genembed(Ys, tau_trials.*(-1), ts_trials)

        # make an in-sample prediction for Y_trial (if needed)
        if error_weights[1]>0
            prediction_insample, ns, temp = insample_prediction(PredictionMethod, Y_trial; samplesize, w, metric, i_cycle=length(τ_vals), kwargs...)
            # compute loss/costs
            costs_insample[i] = compute_costs_from_prediction(PredictionLoss, prediction_insample, Y_trial, PredictionMethod.Tw_in, ns)
            if costs_insample[i] == 0
                println("WARNINNG: The chosen error-metric $PredictionLoss might not be able to work with your data and could yield useless results. Either increase the time series length or choose a different error-metric.")
            end
        end
        # make an out-of-sample prediction for Y_trial (if needed)
        if error_weights[2]>0
            costs_out_of_sample[i] = out_of_sample_prediction(PredictionMethod, PredictionLoss, Y_trial; w, metric, i_cycle=length(τ_vals), kwargs...)
            if costs_out_of_sample[i] == 0
                println("WARNINNG: The chosen error-metric $PredictionLoss might not be able to work with your data and could yield useless results. Either increase the time series length or choose a different error-metric.")
            end
        end
    end
    costs = error_weights[1]*costs_insample .+ error_weights[2]*costs_out_of_sample

    return costs, max_idx, [nothing for i in max_idx]
end


"""
    insample_prediction(pred_meth::AbstractPredictionMethod, Y::AbstractDataset{D, ET};
            K::Int = 3, w::Int = 1, Tw::Int = 1, metric = Euclidean()) → prediction

Compute an in-sample - not iterated - `Tw`-time-steps-ahead prediction of the trajectory `Y::Dataset` ([`Dataset`](@ref)), 
using the prediction method `pred_meth` [`LocalModel`](@ref).

## Keyword arguments
* `K`: Nearest Neighbours
* `w`: Theiler window
* `Tw`: Prediction horizon
* `metric`: Metric for NN search
* `samplesize`: fraction of considered points in the trajectory
* `i_cycle`: Which embedding cycling we are predicting for (for future use)

Note: In case of a local linear prediction method `pred_meth` the number of
nearest neighbours used gets adapted to 2(D+1) - with D the embedding dimension,
if the provided `K` is lower than that number.
"""
function insample_prediction(pred_meth::AbstractLocalPredictionMethod, Y::AbstractDataset{D, ET}; 
    samplesize::Real= 1, w::Int = 1, metric = Euclidean(), i_cycle::Int=1, kwargs...) where {D, ET}

    Tw = pred_meth.Tw_in # total time horizon
    NN = length(Y)-Tw
    if samplesize==1
        ns = 1:NN
        Nfp = length(ns)
    else
        Nfp = Int(floor(samplesize*NN)) # number of considered fiducial points
        ns = sample(vec(1:NN), Nfp, replace = false)  # indices of fiducial points
    end

    prediction = Dataset(zeros(ET, Nfp, D))
    insample_prediction!(pred_meth, Y, prediction, ns; w, K = pred_meth.KNN, Tw, metric)

    return prediction, ns, nothing
end

function insample_prediction!(pred_meth::AbstractLocalPredictionMethod{:zeroth}, Y::AbstractDataset{D, ET}, prediction_new::AbstractDataset{D, ET}, 
            ns::Union{AbstractRange, AbstractVector}; w::Int = 1, metric = Euclidean(), K::Int=1, Tw::Int=1) where {D, ET}

    NN = length(prediction_new)
    vtree = KDTree(Y[1:NN], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, Y[ns], ns, K, w)
    ϵ_ball = zeros(ET, K, D) # preallocation

    # loop over each fiducial point
    for (i,v) in enumerate(ns)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v

        # determine neighborhood one time step ahead
        @inbounds for (k, j) in enumerate(NNidxs)
            ϵ_ball[k, :] .= Y[j+Tw] # consider Tw-step ahead prediction
        end
        # take the average as a prediction
        prediction_new[i] = mean(ϵ_ball; dims=1)
    end
end
function insample_prediction!(pred_meth::AbstractLocalPredictionMethod{:linear}, Y::AbstractDataset{D, ET}, prediction_new::AbstractDataset{D, ET}, 
            ns::Union{AbstractRange, AbstractVector}; w::Int = 1, metric = Euclidean(), K::Int=1, Tw::Int=1) where {D, ET}

    if K < 2*(D+1)
        K = 2*(D+1)
    end
    NN = length(prediction_new)
    vtree = KDTree(Y[1:NN], metric) 
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, Y[ns], ns, K, w)
    prediction = zeros(ET, D) # preallocation
    ϵ_ball = zeros(ET, K, D) # preallocation
    b  = zeros(D) # preallocation
    ar_coeffs = zeros(D, D) # preallocation

    # loop over each fiducial point
    for (i,v) in enumerate(ns)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
        # determine neighborhood one time step ahead
        @inbounds for (k, j) in enumerate(NNidxs)
            ϵ_ball[k, :] .= Y[j + Tw] # consider Tw-step ahead prediction
            A[k,:] = Y[j]
        end
 
        namess = ["X"*string(z) for z = 1:D]
        ee = Meta.parse.(namess)
        formula_expression = Term(:Y) ~ sum(term.(ee))

        for j = 1:D
            data = DataFrame()
            for (cnt,var) in enumerate(namess)
                data[!, var] = A[:,cnt]
            end
            data.Y = ϵ_ball[:,j]

            ols = lm(formula_expression, data)
            b[j] = coef(ols)[1]
            for k = 1:D
                ar_coeffs[j,k] = coef(ols)[k+1]
            end
            prediction[j] = Y[v,:]'*ar_coeffs[j,:] + b[j]
        end
        prediction_new[i] = prediction
    end
end


"""
    out_of_sample_prediction(pred_meth::AbstractLocalPredictionMethod, pred_loss::AbstractPredictionLoss, Y::AbstractDataset{D, ET};
            K::Int = 3, w::Int = 1, Tw::Int = 1, metric = Euclidean()) → average_cost

Compute an out-of-sample `Tw`-time-steps-ahead prediction of the data `Y::Dataset` ([`Dataset`](@ref)) using
the prediction method `pred_meth` [`LocalModel`](@ref).

## Keyword arguments
* `K`: Nearest Neighbours
* `w`: Theiler window
* `Tw`: Prediction horizon
* `metric`: Metric for NN search
* `i_cycle`: Which embedding cycling we are predicting for (for future use)

Note: In case of a local linear prediction method `pred_meth` the number of
nearest neighbours used gets adapted to 2(D+1) - with D the embedding dimension,
if the provided `K` is lower than that number.
"""
function out_of_sample_prediction(pred_meth::AbstractLocalPredictionMethod, pred_loss::AbstractPredictionLoss, 
        Y::AbstractDataset{D, ET}; w::Int = 1, metric = Euclidean(), i_cycle::Int=1, kwargs...) where {D, ET}

    Tw = pred_meth.Tw_out
    num_of_trials = pred_meth.trials
    N = length(Y)
    NN = N-Tw

    # split data into training (90%) and test set (10%)
    N_train = Int(ceil(NN*0.9))
    N_test = NN-N_train

    if num_of_trials ≥ N_test
        println("The test set has not sufficient size to satisfy the given number of out-of-sample trials. This has been adjusted now.")
        num_of_trials = N_test
    end

    ns = sample(vec(N_train+1:N_train+N_test), num_of_trials, replace = false)  # starting indices of trial
    costs = zeros(num_of_trials)
    Threads.@threads for i = 1:num_of_trials
        costs[i] = cost_from_out_of_sample_prediction(pred_meth, pred_loss, Y, ns[i]; w, Tw, metric, kwargs...)
    end
    # return the average of the costs for each trial
    return mean(costs)
end

"""
    Compute costs for the out-of-sample prediction
"""
function cost_from_out_of_sample_prediction(pred_meth::AbstractLocalPredictionMethod{:zeroth}, pred_loss::AbstractPredictionLoss, 
        Y::AbstractDataset{D, ET}, ns::Int; w::Int = 1, Tw::Int = 1, metric = Euclidean(), verbose::Bool=false, kwargs...) where {D, ET}

    predicted = iterated_local_zeroth_prediction(Y[1:ns], pred_meth.KNN, Tw; metric, theiler = w, verbose)
    # compute loss/costs
    return compute_costs_from_prediction(pred_loss, predicted, Y[ns+1:ns+Tw], 0, 1:length(ns+1:ns+Tw))
end
function cost_from_out_of_sample_prediction(pred_meth::AbstractLocalPredictionMethod{:linear}, pred_loss::AbstractPredictionLoss, 
        Y::AbstractDataset{D, ET}, ns::Int; w::Int = 1, Tw::Int = 1, metric = Euclidean(), verbose::Bool=false, kwargs...) where {D, ET}

    predicted = iterated_local_linear_prediction(Y[1:ns], pred_meth.KNN, Tw; metric, theiler = w, verbose)
    # compute loss/costs
    return compute_costs_from_prediction(pred_loss, predicted, Y[ns+1:ns+Tw], 0, 1:length(ns+1:ns+Tw))
end


"""
    Compute the in-sample prediction costs based on the loss-metric determined
    by PredictionLoss
"""
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{1}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int, ns::Union{AbstractRange,AbstractVector}) where {D, T}

    NN = length(ns)
    @assert length(prediction) == length(ns)
    costs = abs2.(Matrix(prediction) - Matrix(Y[ns.+Tw]))
    c = sqrt.(mean(costs; dims=1))
    return c[1]
end
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{2}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int, ns::Union{AbstractRange,AbstractVector}) where {D, T}
    NN = length(ns)
    @assert length(prediction) == length(ns)
    costs = abs2.(Matrix(prediction) - Matrix(Y[ns.+Tw]))
    c = sqrt.(mean(costs; dims=1))
    return mean(c)
end
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{3}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int, ns::Union{AbstractRange,AbstractVector}) where {D, T}
    NN = length(ns)
    @assert length(prediction) == length(ns)
    costs = zeros(D)
    for j = 1:D
        costs[j] = compute_KL_divergence(Vector(prediction[:,j]),Y[ns .+ Tw,j])
    end
    return costs[1]
end
function compute_costs_from_prediction(PredictionLoss::AbstractPredictionLoss{4}, prediction::AbstractDataset{D, T},
                            Y::AbstractDataset{D, T}, Tw::Int, ns::Union{AbstractRange,AbstractVector}) where {D, T}
    NN = length(ns)
    @assert length(prediction) == length(ns)
    costs = zeros(D)
    for j = 1:D
        costs[j] = compute_KL_divergence(Vector(prediction[:,j]),Y[ns .+ Tw,j])
    end
    return mean(costs)
end


"""
    local_zeroth_prediction(Y::Dataset, K::Int = 5; kwargs...) → x_pred, e_expect

Perform a "zeroth" order prediction for the time horizon `Tw` (default = 1) [^Kantz2004]. 
Based on `K` nearest neighbours of the last point of the given trajectory `Y::Dataset` 
([`Dataset`](@ref)), the `Tw`-step ahead prediction is simply the mean of the images of these `K`-nearest
neighbours. 

## Returns
* `x_pred::Vector`: the `Tw`-step ahead prediction vector.
* `e_expect::Vector`: the expected error on the prediction `x_pred`,
computed as the mean of the RMS-errors of all `K`-neighbours-errors.

## Keyword arguments
* `metric = Euclidean()`: Metric used for distance computation
* `theiler::Int = 1`: Theiler window for excluding serially correlated points from
    the nearest neighbour search.
* `Tw::Int = 1`: The prediction time in sampling units. If `Tw > 1`, a multi-step
    prediction is performed.

## References
[^Kantz2004]: Holger Kantz and Thomas Schreiber (2004). [Nonlinear Time Series Analysis. Cambridge nonlinear science series. Cambridge University Press].
"""
function local_zeroth_prediction(Y::Dataset{D,T}, K::Int = 5;
    metric = Euclidean(), theiler::Int = 1, Tw::Int = 1) where {D,T}

    NN = length(Y)
    ns = 1:NN
    vs = Y[ns]
    vtree = KDTree(Y[1:length(Y)-Tw,:], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)

    ϵ_ball = zeros(T, K, D) # preallocation
    # consider NNs of the very last point of the trajectory
    NNidxs = allNNidxs[end] # indices of k nearest neighbors to v
    # determine neighborhood `Tw` time steps ahead
    @inbounds for (i, j) in enumerate(NNidxs)
        ϵ_ball[i, :] .= Y[j + Tw]
    end
    # take the average as a prediction
    prediction = mean(ϵ_ball; dims=1)
    # predicted prediction error
    error_predict = sum((ϵ_ball .- prediction).^2; dims=1)

    return vec(prediction), vec(error_predict)
end

"""
    iterated_local_zeroth_prediction(Y::Dataset, K::Int = 5, Tw::Int = 2; kwargs...) → Y_predict

Perform an iterated one step forecast over `Tw` time steps using the local zeroth
prediction algorithm [`local_zeroth_prediction`](@ref). `Y_predict` is a Vector of 
dimension like `Y`.

## Keyword arguments
* `metric = Euclidean()`: Metric used for distance computation
* `theiler::Int = 1`: Theiler window for excluding serially correlated points from
    the nearest neighbour search.
* `verbose::Bool = false`: When set to `true`, the function prints the actual time
    step, which it is computing.
"""
function iterated_local_zeroth_prediction(Y::Dataset{D,T}, K::Int = 5, Tw::Int = 2;
    metric = Euclidean(), theiler::Int = 1, verbose::Bool = false) where {D,T}

    @assert Tw > 0 "Time horizon must be a positive integer"
    N = length(Y)
    predicted_trajectory = deepcopy(Y)
    for Th = 1:Tw
        if verbose
            println("Compute prediction for time step $Th")
        end
        # iterated one step
        predicted, _ = local_zeroth_prediction(predicted_trajectory, K;
                                            theiler, metric)
        Base.push!(predicted_trajectory, predicted)
    end
    return predicted_trajectory[N+1:end]
end


"""
    local_linear_prediction(Y::Dataset, K::Int = 5; kwargs...) → x_pred, e_expect

Perform a prediction for the time horizon `Tw` (default = 1) by a locally linear
fit [^Farmer1987]. Based on `K` nearest neighbours of the last point of the given trajectory
`Y`, we fit a linear model to these points and their `Tw`-step ahead images. 

## Returns
* `x_pred::Vector`: the `Tw`-step ahead prediction vector.
* `e_expect::Vector`: the expected error on the prediction `x_pred`,
computed as the mean of the RMS-errors of all `K`-neighbours-errors.

## Keywords arguments
* `metric = Euclidean()`: Metric used for distance computation
* `theiler::Int = 1`: Theiler window for excluding serially correlated points from
    the nearest neighbour search.
* `Tw::Int = 1`: The prediction time in sampling units. If `Tw > 1`, a multi-step
    prediction is performed.

## References
[^Farmer1987]: Farmer, J.D., Sidorowich, J.J. (1987). [Predicting chaotic time series. Phys. Rev. Lett 59, 845–848 (1987)](https://doi.org/10.1103/PhysRevLett.59.845).
"""
function local_linear_prediction(Y::Dataset{D,T}, K::Int = 5;
    metric = Euclidean(), theiler::Int = 1, Tw::Int = 1) where {D,T}

    if K < 2*(D+1)
        K = 2*(D+1)
    end

    NN = length(Y)
    ns = 1:NN
    vs = Y[ns]
    vtree = KDTree(Y[1:length(Y)-Tw,:], metric)
    allNNidxs, _ = DelayEmbeddings.all_neighbors(vtree, vs, ns, K, theiler)

    ϵ_ball = zeros(T, K, D) # preallocation
    A = ones(K,D) # datamatrix for later linear equation to solve for AR-process
    # consider NNs of the very last point of the trajectory
    NNidxs = allNNidxs[end] # indices of k nearest neighbors to v

    # determine neighborhood `Tw` time steps ahead
    @inbounds for (i, j) in enumerate(NNidxs)
        ϵ_ball[i, :] .= Y[j + Tw]
        A[i,:] = Y[j]
    end

    # make local linear model of the last point of the trajectory
    prediction = zeros(D)
    b  = zeros(D)
    ar_coeffs = zeros(D, D)
    namess = ["X"*string(i) for i = 1:D]
    ee = Meta.parse.(namess)
    formula_expression = Term(:Y) ~ sum(term.(ee))

    for i = 1:D
        data = DataFrame()
        for (cnt,var) in enumerate(namess)
            data[!, var] = A[:,cnt]
        end
        data.Y = ϵ_ball[:,i]

        ols = lm(formula_expression, data)
        b[i] = coef(ols)[1]
        for j = 1:D
            ar_coeffs[i,j] = coef(ols)[j+1]
        end
        prediction[i] = Y[NN,:]'*ar_coeffs[i,:] + b[i]
    end

    # predicted prediction error
    error_predict = sum((ϵ_ball .- prediction').^2; dims=1)

    return vec(prediction), vec(error_predict)
end

"""
    iterated_local_linear_prediction(Y::Dataset, K::Int = 5, Tw::Int = 2; kwargs...) → Y_predict

Perform an iterated one step forecast over `Tw` time steps using the local linear
prediction algorithm [`local_linear_prediction`](@ref). `Y_predict` is a Vector of dimension like `Y`.

## Keyword arguments
* `metric = Euclidean()`: Metric used for distance computation
* `theiler::Int = 1`: Theiler window for excluding serially correlated points from
    the nearest neighbour search.
* `verbose::Bool = false`: When set to `true`, the function prints the actual time
    step, which it is computing.
"""
function iterated_local_linear_prediction(Y::Dataset{D,T}, K::Int = 5, Tw::Int = 2;
    metric = Euclidean(), theiler::Int = 1, verbose::Bool = false) where {D,T}

    @assert Tw > 0 "Time horizon must be a positive integer"
    N = length(Y)
    predicted_trajectory = deepcopy(Y)
    for Th = 1:Tw
        if verbose
            println("Compute prediction for time step $Th")
        end
        # iterated one step
        predicted, _ = local_linear_prediction(predicted_trajectory, K;
                                            theiler, metric)
        Base.push!(predicted_trajectory, predicted)
    end
    return predicted_trajectory[N+1:end]
end