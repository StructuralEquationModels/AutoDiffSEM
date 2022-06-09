############################################################################################
# constructor for SemZygote types
############################################################################################

function SemZygote(;
        observed::O = SemObservedData,
        imply::I = RAMSymbolicZygote,
        loss::L = SemML,
        optimizer::D = SemOptimizerOptim,
        kwargs...) where {O, I, L, D}

    kwargs = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwargs, observed, imply, loss, optimizer, O, I, D)
    
    observed, imply, loss, optimizer = get_fields!(kwargs, observed, imply, loss, optimizer)

    sem = SemZygote(observed, imply, loss, optimizer)

    return sem
end

function SemForwardDiff(;
        observed::O = SemObservedData,
        imply::I = RAM,
        loss::L = SemML,
        optimizer::D = SemOptimizerOptim,
        kwargs...) where {O, I, L, D}

    kwargs = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwargs, observed, imply, loss, optimizer, O, I, D)
    
    observed, imply, loss, optimizer = get_fields!(kwargs, observed, imply, loss, optimizer)

    sem = SemForwardDiff(observed, imply, loss, optimizer)
    
    return sem
end

##############################################################
# pretty printing
##############################################################

#= function Base.show(io::IO, sem::Sem{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. nameof(typeof(sem.loss.functions))
    print(io, "Sem{$(nameof(O)), $(nameof(I)), $lossfuntypes, $(nameof(D))}")
end =#

function Base.show(io::IO, sem::SemForwardDiff{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "Structural Equation Model : Forward Mode Autodiff\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   imply:       $(nameof(I)) \n")
    print(io, "   optimizer:   $(nameof(D)) \n") 
end

function Base.show(io::IO, sem::SemZygote{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "Structural Equation Model : Zygote Autodiff\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:    $(nameof(O)) \n")
    print(io, "   imply:       $(nameof(I)) \n")
    print(io, "   optimizer:   $(nameof(D)) \n") 
end