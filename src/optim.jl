# General constructors for defining the Optimization Goal
# Further refinements and needed constructors can be found
# in the according folders `./Delay preselection statistics/`
# and `./Cost functions`.

abstract type AbstractMCDTSOptimGoal end

"""
    AbstractDelayPreselection

Supertype of all proposed structs that preselect potential delay values. A DelayPreSelection type has to have a [`get_max_idx`](@ref) and [`get_delay_statistic`](@ref) function attached to it. For examples see the source code of the already implemented [`ContinuityFunction`](@ref) and [`RangeFunction`](@ref)
"""
abstract type AbstractDelayPreselection end

"""
    AbstractLoss

Supertype of all proposed loss/cost functions. A Loss type has to have a [`compute_loss`](@ref) function. Additionally, one can define [`init_embedding_params`](@ref), [`TreeEmbedding.push!`](@ref), [`get_embedding_params_according_to_loss`](@ref). For examples see the source code of the already implemented [`LStatistic`](@ref), [`FNNStatistic`](@ref), [`CCM_ρ`](@ref) and [`PredictionError`](@ref).

All subtypes need to define a function to return a `threshold` for the tolerable `ΔL` decrease for the current
embedding. When `ΔL` exceeds this threshold in an embedding cycle the embedding stops. Note that `ΔL` is a negative value therefore `threshold` must be a small negative number. Per default, the field `treshold` is returned
"""
abstract type AbstractLoss end

"""
  threshold(L::AbstractLoss)

  Returns a `threshold` for the tolerable `ΔL` decrease for the current
  embedding. When `ΔL` exceeds this threshold in an embedding cycle the embedding stops. Note that `ΔL` is a negative value therefore `threshold` must be a small negative number. Per default, the field `treshold` is returned
"""
threshold(L::AbstractLoss) = L.threshold 

"""
    MCDTSOptimGoal <: AbstractMCDTSOptimGoal

Constructor, which handles the loss-/objective function `Γ` and the delay
pre-selection statistic `Λ` MCDTS uses.

## Fieldnames
* `Γ::AbstractLoss`: Chosen loss-function, see the so far available
  [`LStatistic`](@ref) (see also [`uzal_cost`](@ref)), [`FNNStatistic`](@ref), [`CCM_ρ`](@ref) and
  [`PredictionError`](@ref).
* `Λ::AbstractDelayPreselection`: Chosen delay Pre-selection method, see the so
  far available [`ContinuityFunction`](@ref) and [`RangeFunction`](@ref).

## Defaults
* When calling `MCDTSOptimGoal()`, a optimization goal struct is created, which
  uses the [`LStatistic`](@ref) as a loss function `Γ` and the [`ContinuityFunction`](@ref)
  as a delay Pre-selection method Λ.
"""
struct MCDTSOptimGoal <: AbstractMCDTSOptimGoal
    Γ::AbstractLoss
    Λ::AbstractDelayPreselection
end


## Some Defaults for the MCDTSOptimGoal-struct:

# PECUZAL (Continuity statistic + LStatistic)
PecuzalOptim() = MCDTSOptimGoal(LStatistic(), ContinuityFunction())
MCDTSOptimGoal() = PecuzalOptim() # alias
# Continuity & FNN-statistic
FNNOptim() = MCDTSOptimGoal(FNNStatistic(), ContinuityFunction())
# For CCM-causality analysis
CCMOptim() = MCDTSOptimGoal(CCM_ρ(), RangeFunction())
# For prediction with zeroth order predictor and continuity statistic for delay preselection
PredictOptim() = MCDTSOptimGoal(CCM_ρ(), ContinuityFunction())
