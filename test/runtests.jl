using SafeTestsets, Test

const BACKEND_GROUP = uppercase(get(ENV, "BACKEND_GROUP", "All"))
const TESTING_GROUP = uppercase(get(ENV, "TESTING_GROUP", "All"))

macro testif(group, val, expr)
    return quote
        if ($(esc(group)) == "ALL") || $(esc(group)) == uppercase($(esc(val)))
            @testset "$(g)" for g in ($(esc(group)),)
                $(esc(expr))
            end
        end
    end
end

@info "Running tests for $(BACKEND_GROUP) backend"
@info "Running tests for $(TESTING_GROUP) group"

@testset "Complementarity Solve" begin
    @testif TESTING_GROUP "Core" begin
        @testif BACKEND_GROUP "CPU" begin
            @safetestset "Linear Complementarity Problems" begin
                include("core/cpu/lcp.jl")
            end
            @safetestset "Mixed Complementarity Problems" begin
                include("core/cpu/mcp.jl")
            end
        end

        @testif BACKEND_GROUP "CUDA" begin
            @safetestset "Linear Complementarity Problems" begin
                include("core/cuda/lcp.jl")
            end
            @safetestset "Mixed Complementarity Problems" begin
                include("core/cuda/mcp.jl")
            end
        end
    end

    @testif TESTING_GROUP "Applications" begin
        @testif BACKEND_GROUP "CPU" begin
            @testset "Differentiable Controller Learning" begin
                @safetestset "Acrobot with Soft Joint Limits" begin
                    include("applications/cpu/control_learning/soft_joint_acrobot.jl")
                end
                @safetestset "Cart Pole with Soft Walls" begin
                    include("applications/cpu/control_learning/cartpole.jl")
                end
                @safetestset "Partial State Feedback" begin
                    include("applications/cpu/control_learning/partial_state_feedback.jl")
                end
            end
        end
    end

    @testif TESTING_GROUP "QA" begin
        @testif BACKEND_GROUP "CPU" begin
            @safetestset "Aqua Quality Assurance" begin
                include("aqua.jl")
            end
        end
    end
end
