using SafeTestsets, Test

@testset "Complementarity Solve" begin
    @safetestset "Linear Complementarity Problems" begin
        include("linear.jl")
    end
end
