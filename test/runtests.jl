using SafeTestsets, Test

const BACKEND_GROUP = uppercase(get(ENV, "BACKEND_GROUP", "All"))

@testset "Complementarity Solve" begin
    @testset "Core" begin
        if BACKEND_GROUP == "ALL" || BACKEND_GROUP == "CPU"
            @safetestset "Linear Complementarity Problems" begin
                include("core/lcp.jl")
            end
            @safetestset "Mixed Complementarity Problems" begin
                include("core/mcp.jl")
            end
        end
    end

    @testset "Applications" begin
        if BACKEND_GROUP == "ALL" || BACKEND_GROUP == "CPU"
            @testset "Differentiable Controller Learning" begin
                @safetestset "Acrobot with Soft Joint Limits" begin
                    include("applications/control_learning/soft_joint_acrobot.jl")
                end
                @safetestset "Cart Pole with Soft Walls" begin
                    include("applications/control_learning/cartpole.jl")
                end
                @safetestset "Partial State Feedback" begin
                    include("applications/control_learning/partial_state_feedback.jl")
                end
            end
        end
    end

    if BACKEND_GROUP == "ALL" || BACKEND_GROUP == "CPU"
        @safetestset "Aqua Quality Assurance" begin
            include("core/aqua.jl")
        end
    end
end
