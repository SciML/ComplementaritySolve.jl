## Works only if M is positive definite and symmetric
@concrete struct BokhovenIterativeAlgorithm <: AbstractComplementarityAlgorithm
    nlsolver
end

BokhovenIterativeAlgorithm() = BokhovenIterativeAlgorithm(NewtonRaphson())

@truncate_stacktrace BokhovenIterativeAlgorithm

## NOTE: It is a steady state problem so we could in-principle use an ODE Solver
for batched in (true, false)
    @eval @views function __solve(prob::LinearComplementarityProblem{iip, $batched},
        alg::BokhovenIterativeAlgorithm,
        u0,
        M,
        q;
        kwargs...) where {iip}
        A = I➕x⁻¹(M)
        B = matmul(A, I➖x(M))
        b = -matmul(A, q)

        θ = vcat(vec(B), vec(b))

        _get_B(θ) = reshape(view(θ, 1:length(B)), size(B))
        _get_b(θ) = reshape(view(θ, (length(B) + 1):length(θ)), size(b))

        if iip
            function objective!(residual, u, θ)
                residual .= _get_b(θ) .- u
                matmul!(residual, _get_B(θ), abs.(u), true, true)
                return residual
            end

            _prob = NonlinearProblem(NonlinearFunction{true}(objective!), u0, θ)
        else
            objective(u, θ) = matmul(_get_B(θ), abs.(u)) .+ _get_b(θ) .- u

            _prob = NonlinearProblem(NonlinearFunction{false}(objective), u0, θ)
        end
        sol = solve(_prob, alg.nlsolver; kwargs...)

        z = abs.(sol.u) .+ sol.u

        return LinearComplementaritySolution(z, sol.resid, prob, alg, sol.retcode)
    end
end
