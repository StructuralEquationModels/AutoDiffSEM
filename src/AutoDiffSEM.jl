module AutoDiffSEM

    using StructuralEquationModels, Zygote, ChainRulesCore, Symbolics, SparseArrays, LinearAlgebra
    import StructuralEquationModels:
        objective!, gradient!, hessian!, 
        objective_gradient!, objective_hessian!, gradient_hessian!,
        objective_gradient_hessian!,
        get_Σ_symbolic_RAM, get_μ_symbolic_RAM, 
        non_posdef_return, Σ, obs_cov,
        SemImplySymbolic,
        set_field_type_kwargs!, get_fields!, get_SemLoss, set_RAMConstants!, fill_A_S_M,
        identifier,
        n_par, update_observed, Σ, ∇Σ, μ, ∇μ, Σ_function, ∇Σ_function, has_meanstructure
    # import StructuralEquationModels: print_type_name, print_field_types, sem_fit, start_val

    include("types.jl")
    include("objective_gradient_hessian.jl")
    include("imply/symbolic_zygote.jl")

    include("loss/ML.jl")

    include("frontend/specification/Sem.jl")

    export SemForwardDiff, SemZygote, RAMSymbolicZygote
        #objective!, gradient!, hessian!, 
        #objective_gradient!, objective_hessian!, gradient_hessian!,
        #objective_gradient_hessian!
end