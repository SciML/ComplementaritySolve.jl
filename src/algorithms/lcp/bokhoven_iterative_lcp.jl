## Works only if M is positive definite and symmetric
@concrete struct BokhovenIterativeLCPAlgorithm <: AbstractComplementarityAlgorithm
    nlsolver
end

BokhovenIterativeLCPAlgorithm() = BokhovenIterativeLCPAlgorithm(NewtonRaphson())

@truncate_stacktrace BokhovenIterativeLCPAlgorithm

## NOTE: It is a steady state problem so we could in-principle use an ODE Solver
function solve(prob::LinearComplementarityProblem{iip, false},
    alg::BokhovenIterativeLCPAlgorithm,
    args...;
    kwargs...) where {iip}
    A = pinv(I + prob.M)
    B = A * (I - prob.M)
    b = -A * prob.q

    θ = vcat(vec(B), b)

    _get_B(θ) = reshape(view(θ, 1:length(B)), size(B))
    _get_b(θ) = view(θ, (length(B) + 1):length(θ))

    if iip
        function objective!(residual, u, θ)
            mul!(residual, _get_B(θ), abs.(u))
            residual .+= _get_b(θ) .- u
            return residual
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(objective!), prob.u0, θ)
    else
        objective(u, θ) = _get_B(θ) * abs.(u) .+ _get_b(θ) .- u

        _prob = NonlinearProblem(NonlinearFunction{false}(objective), prob.u0, θ)
    end
    sol = solve(_prob, alg.nlsolver; kwargs...)

    z = abs.(sol.u)
    b .= z .- sol.u # |u| - u # This is `w`. Just reusing `b` to save memory
    z .+= sol.u     # |u| + u

    return LinearComplementaritySolution(z, b, sol.resid, prob, alg)
end
