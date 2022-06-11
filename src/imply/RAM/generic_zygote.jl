############################################################################################
### Types
############################################################################################
mutable struct RAMZygote{A1, A2, A3, A4, A5, A6, V, V2, I1, I2, I3, M1, M2, M3, M4, S1, S2, S3, B, D} <: SemImply
    Σ::A1
    A::A2
    S::A3
    F::A4
    μ::A5
    M::A6

    n_par::V
    ram_matrices::V2
    has_meanstructure::B

    A_indices::I1
    S_indices::I2
    M_indices::I3

    F⨉I_A⁻¹::M1
    F⨉I_A⁻¹S::M2
    I_A::M3
    I_A⁻¹::M4

    ∇A::S1
    ∇S::S2
    ∇M::S3

    identifier::D
end

using StructuralEquationModels

############################################################################################
### Constructors
############################################################################################

function RAMZygote(;
        specification,
        #vech = false,
        gradient = true,
        meanstructure = false,
        kwargs...)

    ram_matrices = RAMMatrices(specification)
    identifier = StructuralEquationModels.identifier(ram_matrices)


    # get dimensions of the model
    n_par = length(ram_matrices.parameters)
    n_var, n_nod = ram_matrices.size_F
    parameters = ram_matrices.parameters
    F = zeros(ram_matrices.size_F); F[CartesianIndex.(1:n_var, ram_matrices.F_ind)] .= 1.0

    # get indices
    A_indices = copy(ram_matrices.A_ind)
    S_indices = copy(ram_matrices.S_ind)
    !isnothing(ram_matrices.M_ind) ? M_indices = copy(ram_matrices.M_ind) : M_indices = nothing

    #preallocate arrays
    A_pre = zeros(n_nod, n_nod)
    S_pre = zeros(n_nod, n_nod)
    !isnothing(M_indices) ? M_pre = zeros(n_nod) : M_pre = nothing

    set_RAMConstants!(A_pre, S_pre, M_pre, ram_matrices.constants)
    
    A_pre = check_acyclic(A_pre, n_par, A_indices)

    # pre-allocate some matrices
    Σ = zeros(n_var, n_var)
    F⨉I_A⁻¹ = zeros(n_var, n_nod)
    F⨉I_A⁻¹S = zeros(n_var, n_nod)
    I_A = similar(A_pre)

    if gradient
        ∇A = get_matrix_derivative(A_indices, parameters, n_nod^2)
        ∇S = get_matrix_derivative(S_indices, parameters, n_nod^2)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if meanstructure

        has_meanstructure = Val(true)

        if gradient
            ∇M = get_matrix_derivative(M_indices, parameters, n_nod)
        else
            ∇M = nothing
        end

        μ = zeros(n_var)

    else
        has_meanstructure = Val(false)
        M_indices = nothing
        M_pre = nothing
        μ = nothing
        ∇M = nothing
    end

    return RAMZygote(
        Σ,
        A_pre,
        S_pre,
        F,
        μ,
        M_pre,

        n_par,
        ram_matrices,
        has_meanstructure,

        A_indices,
        S_indices,
        M_indices,

        F⨉I_A⁻¹,
        F⨉I_A⁻¹S,
        I_A,
        copy(I_A),

        ∇A,
        ∇S,
        ∇M,

        identifier
    )
end

############################################################################################
### methods
############################################################################################

# dispatch on meanstructure
objective!(imply::RAMZygote, par, model::AbstractSemSingle) = 
    objective!(imply, par, model, imply.has_meanstructure)

# objective and gradient
function objective!(imply::RAMZygote, parameters, model, has_meanstructure::Val{T}) where T

    imply.A = fill_matrix_wrap(imply.A, imply.A_indices, parameters, imply.∇A)
    imply.S = fill_matrix_wrap(imply.S, imply.S_indices, parameters, imply.∇S)

    F⨉I_A⁻¹ = imply.F*inv(I-imply.A)

    imply.Σ = F⨉I_A⁻¹*imply.S*F⨉I_A⁻¹'

    if T
        imply.M = fill_matrix_wrap(imply.M, imply.M_indices, parameters, imply.∇M)
        imply.μ = F⨉I_A⁻¹*imply.M
    end

end

############################################################################################
### Recommended methods
############################################################################################

identifier(imply::RAMZygote) = imply.identifier
n_par(imply::RAMZygote) = imply.n_par

function update_observed(imply::RAMZygote, observed::SemObserved; kwargs...) 
    if n_man(observed) == size(imply.Σ, 1)
        return imply
    else
        return RAM(;observed = observed, kwargs...)
    end
end

############################################################################################
### additional methods
############################################################################################

Σ(imply::RAMZygote) = imply.Σ
μ(imply::RAMZygote) = imply.μ

A(imply::RAMZygote) = imply.A
S(imply::RAMZygote) = imply.S
F(imply::RAMZygote) = imply.F
M(imply::RAMZygote) = imply.M

∇A(imply::RAMZygote) = imply.∇A
∇S(imply::RAMZygote) = imply.∇S
∇M(imply::RAMZygote) = imply.∇M

A_indices(imply::RAMZygote) = imply.A_indices
S_indices(imply::RAMZygote) = imply.S_indices
M_indices(imply::RAMZygote) = imply.M_indices

F⨉I_A⁻¹(imply::RAMZygote) = imply.F⨉I_A⁻¹
F⨉I_A⁻¹S(imply::RAMZygote) = imply.F⨉I_A⁻¹S
I_A(imply::RAMZygote) = imply.I_A
I_A⁻¹(imply::RAMZygote) = imply.I_A⁻¹ # only for gradient available!

has_meanstructure(imply::RAMZygote) = imply.has_meanstructure

ram_matrices(imply::RAMZygote) = imply.ram_matrices

############################################################################################
### additional functions
############################################################################################

function fill_matrix_wrap(M_pre, M_indices, parameters, ∇M)
    M = copy(M_pre)
    fill_matrix(M, M_indices, parameters)
    return M
end

function ChainRulesCore.rrule(::typeof(fill_matrix_wrap), M_pre, M_indices, parameters, ∇M)
    M = fill_matrix_wrap(M_pre, M_indices, parameters, ∇M)

    function fill_matrix_wrap_pullback(ȳ)
        f̄ = NoTangent()
        M̄_pre = NoTangent()
        M̄_indices = NoTangent()
        p̄ar = @thunk(∇M'*vec(ȳ))
        ∇M̄ = NoTangent()
        return f̄, M̄_pre, M̄_indices, p̄ar, ∇M̄
    end

    return M, fill_matrix_wrap_pullback
end

function check_acyclic(A_pre, n_par, A_indices)
    # fill copy of A-matrix with random parameters
    A_rand = copy(A_pre)
    randpar = rand(n_par)

    fill_matrix(
        A_rand,
        A_indices,
        randpar)

    # check if the model is acyclic
    acyclic = isone(det(I-A_rand))

    # check if A is lower or upper triangular
    if iszero(A_rand[.!tril(ones(Bool, size(A_pre)...))])
        A_pre = LowerTriangular(A_pre)
    elseif iszero(A_rand[.!tril(ones(Bool, size(A_pre)...))'])
        A_pre = UpperTriangular(A_pre)
    elseif acyclic
        @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.\n" maxlog=1
    end

    return A_pre
end