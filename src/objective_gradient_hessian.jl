############################################################################################
# methods for ForwardDiff
############################################################################################

gradient!(gradient, model::SemForwardDiff, par, has_gradient::Val{false}) =
    ForwardDiff.gradient!(gradient, x -> objective!(model, x), par)

hessian!(hessian, model::SemForwardDiff, par) = 
    ForwardDiff.hessian!(hessian, x -> objective!(model, x), par)

function objective_gradient!(gradient, model::SemForwardDiff, parameters)
    fill!(gradient, zero(eltype(gradient)))
    gradient!(gradient, model, parameters)
    return objective!(model, parameters)
end

############################################################################################
# methods for Zygote
############################################################################################

function gradient!(gradient, model::SemZygote, par)
    grad = Zygote.gradient(x -> objective!(model, x), par)[1]
    if !isnothing(grad)
        gradient .= grad
    else
        fill!(gradient, one(eltype(gradient)))
    end
end

function hessian!(hessian, model::SemZygote, par)
    hessian .= Zygote.hessian(x -> objective!(model, x), par)
end

function objective_gradient!(gradient, model::SemZygote, parameters)
    ob, grad = Zygote.withgradient(x -> objective!(model, x), parameters)
    if !isnothing(grad[1])
        gradient .= grad[1]
    else
        fill!(gradient, one(eltype(gradient)))
    end
    return ob
end

############################################################################################
# shared methods
############################################################################################

# other methods
function gradient_hessian!(
        gradient,
        hessian,
        model::Union{SemZygote, SemForwardDiff},
        parameters)
    gradient!(gradient, model, parameters)
    hessian!(hessian, model, parameters)
end

function objective_hessian!(hessian, model::Union{SemZygote, SemForwardDiff}, parameters)
    hessian!(hessian, model, parameters)
    return objective!(model, parameters)
end

function objective_gradient_hessian!(
        gradient, 
        hessian, 
        model::Union{SemZygote, SemForwardDiff}, 
        parameters)
    hessian!(hessian, model, parameters)
    return objective_gradient!(gradient, model, parameters)
end