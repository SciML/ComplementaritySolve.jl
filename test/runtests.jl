using SafeTestsets, Test

@testset "Complementarity Solve" begin
    @testset "Complementarity Problems" begin
        @safetestset "Linear Complementarity Problems" begin
            include("core/lcp.jl")
        end
        @safetestset "Mixed Complementarity Problems" begin
            include("core/mcp.jl")
        end
    end

    @safetestset "Aqua Quality Assurance" begin
        include("core/aqua.jl")
    end
end
