using Test, SafeTestsets

@safetestset "Political Democracy Example" begin
    include("political_democracy/political_democracy.jl")
end

@safetestset "Multigroup Example" begin
    include("multigroup/multigroup.jl")
end