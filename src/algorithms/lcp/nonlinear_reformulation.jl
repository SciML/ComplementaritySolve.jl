
for method in (:minmax, :smooth)
    algType = NonlinearReformulation{method}
    op = Symbol("$(method)_transform")

    @eval function solve(prob::LinearComplementarityProblem{true, false},
        alg::$algType;
        kwargs...)
        function f!(out, u, θ)
            M = reshape(view(θ, 1:length(prob.M)), size(prob.M))
            q = view(θ, (length(prob.M) + 1):length(θ))
            out .= q
            mul!(out, M, u, true, true)
            return out
        end

        function objective!(residual, u, θ)
            f!(residual, u, θ)
            residual .= $(op).(residual, u)
            return residual
        end

        θ = vcat(vec(prob.M), prob.q)
        _prob = NonlinearProblem(NonlinearFunction{true}(objective!), prob.u0, θ)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        z = sol.u
        w = f!(similar(z), z, θ)

        return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
    end

    # Batched In-place
    @eval function solve(prob::LinearComplementarityProblem{true, true},
        alg::$algType;
        kwargs...)
        function f!(out, u, θ)
            M = reshape(view(θ, 1:length(prob.M)), size(prob.M))
            out .= reshape(view(θ, (length(prob.M) + 1):length(θ)), size(prob.q, 1), 1, :)
            batched_mul!(out, M, reshape(u, size(u, 1), 1, :), true, true)
            return out
        end

        function objective!(residual, u, θ)
            f!(residual, u, θ)
            residual .= $(op).(residual, u)
            return residual
        end

        θ = vcat(vec(prob.M), vec(prob.q))
        _prob = NonlinearProblem(NonlinearFunction{true}(objective!),
            reshape(prob.u0, size(prob.u0, 1), 1, :),
            θ)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        z = sol.u
        w = f!(similar(z), z, θ)

        return LinearComplementaritySolution(dropdims(z; dims=2),
            dropdims(w; dims=2),
            sol.resid,
            prob,
            alg)
    end

    # Unbatched Out-of-place
    @eval function solve(prob::LinearComplementarityProblem{false, false},
        alg::$algType;
        kwargs...)
        function f(u, θ)
            M = reshape(view(θ, 1:length(prob.M)), size(prob.M))
            q = view(θ, (length(prob.M) + 1):length(θ))
            return M * u .+ q
        end

        objective(u, θ) = $(op).(f(u, θ), u)

        θ = vcat(vec(prob.M), prob.q)
        _prob = NonlinearProblem(NonlinearFunction{false}(objective), prob.u0, θ)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        z = sol.u
        w = f(z, θ)

        return LinearComplementaritySolution(z, w, sol.resid, prob, alg)
    end

    # Batched In-place
    @eval function solve(prob::LinearComplementarityProblem{false, true},
        alg::$algType;
        kwargs...)
        function f(u, θ)
            M = reshape(view(θ, 1:length(prob.M)), size(prob.M))
            q = reshape(view(θ, (length(prob.M) + 1):length(θ)), size(prob.q, 1), 1, :)
            return M ⊠ u .+ q
        end

        objective(u, θ) = $(op).(f(u, θ), u)

        θ = vcat(vec(prob.M), vec(prob.q))
        _prob = NonlinearProblem(NonlinearFunction{false}(objective),
            reshape(prob.u0, size(prob.u0, 1), 1, :),
            θ)
        sol = solve(_prob, alg.nlsolver; kwargs...)

        z = sol.u
        w = f(z, θ)

        return LinearComplementaritySolution(dropdims(z; dims=2),
            dropdims(w; dims=2),
            sol.resid,
            prob,
            alg)
    end
end
