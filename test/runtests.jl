using Test, SafeTestsets

@safetestset "SemForwardDiff" begin include("SemForwardDiff/forwarddiff.jl") end 
@safetestset "SemZygote" begin include("SemZygote/zygote.jl") end