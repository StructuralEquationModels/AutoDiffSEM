using AutoDiffSEM, StructuralEquationModels, Test, FiniteDiff

include("../helper.jl")

spec = RAMMatrices(nothing, nothing, nothing, nothing, [:hi, :yo], nothing, nothing, nothing)

model = AutoDiffSEM.SemForwardDiff(
    observed = SemObservedData(specification = nothing, data = rand(10,10)),
    imply = ImplyEmpty(specification = spec),
    loss = SemRidge(which_ridge = [1, 2], α_ridge = 0.2, n_par = 2)
)

par = [2.3, 2.5]


@testset "SemForwardDiff | objective, gradient, hessian" begin
    test_gradient(model, par; rtol = 1e-9)
end

@testset "SemForwardDiff | solution" begin
    fit = sem_fit(model)
    @test solution(fit) ≈ [0.0, 0.0]
end

obj_true = 0.2*sum(par.^2)
grad = [0.0, 0.0]
grad_true = [0.2*2*2.3, 0.2*2*2.5]

@testset "SemForwardDiff" begin
    @test objective!(model, par) == obj_true
    gradient!(grad, model, par)
    @test grad == grad_true
    grad .= 0.0
    @test objective_gradient!(grad, model, par) == obj_true
    @test grad == grad_true
    grad .= 0.0
    @test objective_gradient!(grad, model, par) == obj_true
end