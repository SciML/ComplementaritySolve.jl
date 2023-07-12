using SafeTestsets, Test

@testset "Complementarity Solve" begin
    @testset "Core" begin
        @safetestset "Linear Complementarity Problems" begin
            include("core/lcp.jl")
        end
        @safetestset "Mixed Complementarity Problems" begin
            include("core/mcp.jl")
        end
    end

    @testset "Applications" begin
        @testset "Differentiable Controller Learning" begin
            @safetestset "Acrobot with Soft Joint Limits" begin
                include("applications/control_learning/soft_joint_acrobot.jl")
            end
        end
    end

    @safetestset "Aqua Quality Assurance" begin
        include("core/aqua.jl")
    end
end
