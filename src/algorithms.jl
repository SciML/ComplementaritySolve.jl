abstract type AbstractComplementarityAlgorithm end

# Linear Complementarity Problem
## Works only if M is positive definite and symmetric
@kwdef struct BokhovenIterativeLCPAlgorithm{S} <: AbstractComplementarityAlgorithm
    nlsolver::S = NewtonRaphson()
end

## NOTE: It is a steady state problem so we could in-principle use an ODE Solver
function solve(prob::LinearComplementarityProblem, alg::BokhovenIterativeLCPAlgorithm,
               args...; kwargs...)
    A = pinv(I + prob.M)
    B = A * (I - prob.M)
    b = -A * prob.q

    θ = vcat(vec(B), b)

    function objective(u, θ)
        return view(θ, (length(B) + 1):length(θ)) .+
               reshape(view(θ, 1:length(B)), size(B)) * abs.(u) .- u
    end

    _prob = NonlinearProblem(NonlinearFunction{false}(objective), prob.u0, θ)
    sol = solve(_prob, alg.nlsolver, args...; kwargs...)

    w = abs.(sol.u) .- sol.u
    z = sol.u .+ abs.(sol.u)

    return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
end

# Reformulate Problems as a Nonlinear Problem
struct NonlinearReformulation{method, S} <: AbstractComplementarityAlgorithm
    nlsolver::S
end

function NonlinearReformulation(method::Symbol=:smooth, nlsolver=NewtonRaphson())
    return NonlinearReformulation{method, typeof(nlsolver)}(nlsolver)
end

for method in (:minmax, :smooth)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")
    @eval function solve(prob::MixedComplementarityProblem{true}, alg::$algType, args...;
                         kwargs...)
        function f!(residual, u, θ)
            prob.f(residual, u, θ)
            residual .= $(op).(residual, u, prob.lb, prob.ub)
            return residual
        end

        _prob = NonlinearProblem(NonlinearFunction{true}(f!), prob.u0, prob.p)
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        return sol
    end

    @eval function solve(prob::MixedComplementarityProblem{false}, alg::$algType, args...;
                         kwargs...)
        f(u, θ) = $(op).(prob.f(u, θ), u, prob.lb, prob.ub)

        _prob = NonlinearProblem(NonlinearFunction{false}(f), prob.u0, prob.p)
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        return sol
    end

    @eval function solve(prob::LinearComplementarityProblem, alg::$algType, args...;
                         kwargs...)
        function f(u, θ)
            M = reshape(view(θ, 1:length(prob.M)), size(prob.M))
            q = view(θ, (length(prob.M) + 1):length(θ))
            return M * u .+ q
        end

        residual(u, θ) = $(op).(f(u, θ), u)

        θ = vcat(vec(prob.M), prob.q)
        _prob = NonlinearProblem(NonlinearFunction{false}(residual), prob.u0, θ)
        sol = solve(_prob, alg.nlsolver, args...; kwargs...)

        z = sol.u
        w = f(z, θ)

        return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
    end
end
