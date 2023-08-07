@concrete struct InteriorPointMethod <: AbstractComplementarityAlgorithm
    linsolve
    η
end

function InteriorPointMethod(linsolve=nothing; tolerance=1.0f-5)
    return InteriorPointMethod(linsolve, tolerance)
end

@truncate_stacktrace InteriorPointMethod

function __feasible_steplength(x, Δx, cache; dims=:)
    T = eltype(x)
    cache .= x ./ (Δx .+ eps(T))
    η = minimum(zᵢ -> ifelse(zᵢ ≥ 0, T(Inf), -zᵢ), cache; dims)
    return min(T(0.999) * η, T(1))
end

function __make_ipm_linsolve_operator(M, zₖ, wₖ, Δzw, ::Val{batched}) where {batched}
    L = size(zₖ, 1)
    @views function matvec(v, u, p, t)
        if batched
            v = reshape(v, 2L, :)
            u = reshape(u, 2L, :)
        end
        Δz, Δw = selectdim(u, 1, 1:L), selectdim(u, 1, (L + 1):(2L))
        selectdim(v, 1, 1:L) .= Δw
        matmul!(selectdim(v, 1, 1:L), M, Δz, true, -1)
        selectdim(v, 1, (L + 1):(2L)) .= zₖ .* Δw .+ wₖ .* Δz
        return vec(v)
    end
    return FunctionOperator(matvec, Δzw)
end

## For details see https://sites.math.washington.edu/~burke/crs/408f/notes/lcp/lcp.pdf
for batched in (true, false)
    @eval @views function __solve(prob::LinearComplementarityProblem{iip, $batched},
        alg::InteriorPointMethod,
        u0,
        M,
        q;
        maxiters=1000,
        abstol=nothing,
        reltol=nothing,
        kwargs...) where {iip}
        @assert abstol === nothing&&reltol === nothing "Use the tolerance keyword argument \
                                                        while Solver construction instead"
        T = eltype(u0)
        σ = zero(T)
        L, N = size(u0, 1), 1
        $(batched) && (N = max(size(u0, 2), size(q, 2), size(M, 3)))
        η = alg.η

        # Can't use zero initialization
        z = iszero(u0) ? fill!(u0, T(2)) : u0

        α_cache = similar(z)
        if $(batched)
            w_cache = similar(z, L, N)
        else
            w_cache = copy(q)
        end
        matmul!(w_cache, M, z, true, true)
        w = min.(w_cache, T(2))

        τ = sum(z .* w)
        $(batched) && (τ /= N)
        ρ = norm(w_cache, Inf)

        # Setup Linsolve
        Δzw = similar(z, 2L, N)
        A_ = __make_ipm_linsolve_operator(M, z, w, vec(Δzw), Val($batched))
        b_ = similar(z, 2L, N)
        lincache = init(LinearProblem(A_, vec(b_); u0=vec(Δzw)), alg.linsolve; kwargs...)

        iter = 1
        while τ > η && ρ > η && iter ≤ maxiters
            lincache.A = __make_ipm_linsolve_operator(M, z, w, vec(Δzw), Val($batched))
            selectdim(b_, 1, 1:L) .= w .- w_cache
            selectdim(b_, 1, (L + 1):(2L)) .= σ * τ .- z .* w
            lincache.b = vec(b_)
            solve!(lincache)

            Δz, Δw = Δzw[1:L, :], Δzw[(L + 1):(2L), :]

            η₁ = __feasible_steplength(z, Δz, α_cache)
            η₂ = __feasible_steplength(w, Δw, α_cache)

            z .+= η₁ .* Δz
            w .+= η₂ .* Δw

            τ = sum(z .* w)
            $(batched) && (τ /= N)
            w_cache .= q
            matmul!(w_cache, M, z, true, true)
            ρ = norm(w_cache, Inf)

            σ = ifelse(τ ≤ η && ρ > η,
                T(1),
                min(T(0.5), (1 - η₁)^2, (1 - η₂)^2, abs(ρ - τ) / (ρ + 10 * τ)))

            iter += 1
        end

        retcode = ifelse(iter == maxiters, ReturnCode.MaxIters, ReturnCode.Success)
        return LinearComplementaritySolution(z, [τ, ρ], prob, alg, retcode)
    end
end
