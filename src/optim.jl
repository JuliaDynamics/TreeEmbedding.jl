# General constructors for defining the Optimization Goal
# Further refinements and needed constructors can be found
# in the according folders `./Delay preselection statistics/`
# and `./Cost functions`.

abstract type AbstractMCDTSOptimGoal end

"""
    AbstractDelayPreselection

Supertype of all proposed structs that preselect potential delay values. A DelayPreSelection type has to have a [`get_max_idx`](@ref) and [`get_delay_statistic`](@ref) function attached to it. For examples see the source code of the already implemented [`Continuity_function`](@ref) and [`Range_function`](@ref)
"""
abstract type AbstractDelayPreselection end

"""
    AbstractLoss

Supertype of all proposed loss/cost functions. A Loss type has to have a [`compute_loss`](@ref) function. Additionally, one can define [`init_embedding_params`](@ref), [`TreeEmbedding.push!`](@ref), [`get_embedding_params_according_to_loss`](@ref). For examples see the source code of the already implemented [`L_statistic`](@ref), [`FNN_statistic`](@ref), [`CCM_ρ`](@ref) and [`Prediction_error`](@ref).
"""
abstract type AbstractLoss end

"""
    MCDTSOptimGoal <: AbstractMCDTSOptimGoal

Constructor, which handles the loss-/objective function `Γ` and the delay
pre-selection statistic `Λ` MCDTS uses.

## Fieldnames
* `Γ::AbstractLoss`: Chosen loss-function, see the so far available
  [`L_statistic`](@ref) (see also [`uzal_cost`](@ref)), [`FNN_statistic`](@ref), [`CCM_ρ`](@ref) and
  [`Prediction_error`](@ref).
* `Λ::AbstractDelayPreselection`: Chosen delay Pre-selection method, see the so
  far available [`Continuity_function`](@ref) and [`Range_function`](@ref).

## Defaults
* When calling `MCDTSOptimGoal()`, a optimization goal struct is created, which
  uses the [`L_statistic`](@ref) as a loss function `Γ` and the [`Continuity_function`](@ref)
  as a delay Pre-selection method Λ.
"""
struct MCDTSOptimGoal <: AbstractMCDTSOptimGoal
    Γ::AbstractLoss
    Λ::AbstractDelayPreselection
end


## Some Defaults for the MCDTSOptimGoal-struct:

# PECUZAL (Continuity statistic + L_statistic)
PecuzalOptim() = MCDTSOptimGoal(L_statistic(), Continuity_function())
MCDTSOptimGoal() = PecuzalOptim() # alias
# Continuity & FNN-statistic
FNNOptim() = MCDTSOptimGoal(FNN_statistic(), Continuity_function())
# For CCM-causality analysis
CCMOptim() = MCDTSOptimGoal(CCM_ρ(), Range_function())
# For prediction with zeroth order predictor and continuity statistic for delay preselection
PredictOptim() = MCDTSOptimGoal(CCM_ρ(), Continuity_function())
