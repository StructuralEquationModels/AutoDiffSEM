## Test RAMSymbolicZ
using AutoDiffSEM, StructuralEquationModels

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    ind60 → dem60
    dem60 → dem65
    ind60 → dem65

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

end

partable = ParameterTable(
    latent_vars = latent_vars,
    observed_vars = observed_vars,
    graph = graph)

data = example_data("political_democracy")

model_an = Sem(
    specification = partable,
    data = data
)

model_fit_an = sem_fit(model_an)

sv = start_val(model_an)

model = SemZygote(
    specification = partable,
    data = data
)

objective!(model, sv) ≈ objective!(model_an, sv)

grad = similar(sv)
grad_an = similar(sv)

gradient!(grad, model, sv)
gradient!(grad_an, model_an, sv)

grad ≈ grad_an

model_fit = sem_fit(model; start_val = sv)

maximum(abs.(solution(model_fit) - solution(model_fit_an))) < 1e-2

using BenchmarkTools

@benchmark gradient!($grad, $model, $sv)

@benchmark gradient!($grad_an, $model_an, $sv)

@benchmark objective!($model, $sv)

@benchmark objective!($model_an, $sv)