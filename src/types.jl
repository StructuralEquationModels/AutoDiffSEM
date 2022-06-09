"""
    SemForwardDiff(;observed = SemObservedData, imply = RAM, loss = SemML, optimizer = SemOptimizerOptim, kwargs...)

Constructor for `SemForwardDiff`.
All additional kwargs are passed down to the constructors for the observed, imply, loss and optimizer fields.

# Arguments
- `observed`: object of subtype `SemObserved` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `optimizer`: object of subtype `SemOptimizer` or a constructor.

Returns a Sem with fields
- `observed::SemObserved`: Stores observed data, sample statistics, etc. See also [`SemObserved`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `optimizer::SemOptimizer`: Connects the model to the optimizer. See also [`SemOptimizer`](@ref).
"""
struct SemForwardDiff{O <: SemObserved, I <: SemImply, L <: SemLoss, D <: SemOptimizer} <: AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I 
    loss::L 
    optimizer::D
end

"""
    SemZygote(;observed = SemObservedData, imply = RAM, loss = SemML, optimizer = SemOptimizerOptim, kwargs...)

Constructor for `SemZygote`.
All additional kwargs are passed down to the constructors for the observed, imply, loss and optimizer fields.

# Arguments
- `observed`: object of subtype `SemObserved` or a constructor.
- `imply`: object of subtype `SemImply` or a constructor.
- `loss`: object of subtype `SemLossFunction`s or constructor; or a tuple of such.
- `optimizer`: object of subtype `SemOptimizer` or a constructor.

Returns a Sem with fields
- `observed::SemObserved`: Stores observed data, sample statistics, etc. See also [`SemObserved`](@ref).
- `imply::SemImply`: Computes model implied statistics, like Σ, μ, etc. See also [`SemImply`](@ref).
- `loss::SemLoss`: Computes the objective and gradient of a sum of loss functions. See also [`SemLoss`](@ref).
- `optimizer::SemOptimizer`: Connects the model to the optimizer. See also [`SemOptimizer`](@ref).
"""
struct SemZygote{O <: SemObserved, I <: SemImply, L <: SemLoss, D <: SemOptimizer} <: AbstractSemSingle{O, I, L, D}
    observed::O
    imply::I 
    loss::L 
    optimizer::D
end