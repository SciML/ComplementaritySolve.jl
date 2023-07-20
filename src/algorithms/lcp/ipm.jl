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

function __make_ipm_linsolve_operator(M::AbstractMatrix,
    zₖ::AbstractVector,
    wₖ::AbstractVector,
    Δzw::AbstractVector)
    L = length(zₖ)
    @views function matvec(v::AbstractVector, u::AbstractVector, p, t)
        Δz, Δw = u[1:L], u[(L + 1):(2L)]
        v[1:L] .= M * Δz .- Δw
        v[(L + 1):(2L)] .= zₖ .* Δw .+ wₖ .* Δz
        return v
    end
    return FunctionOperator(matvec, Δzw)
end

## For details see https://sites.math.washington.edu/~burke/crs/408f/notes/lcp/lcp.pdf
@views function __solve(prob::LinearComplementarityProblem{iip, false},
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
    N = length(u0)
    η = alg.η

    # Can't use zero initialization
    z = iszero(u0) ? fill!(u0, T(2)) : u0

    α_cache = similar(z)
    w_cache = copy(q)
    mul!(w_cache, M, z, true, true)
    w = min.(w_cache, T(2))

    τ = (z' * w)
    ρ = norm(w_cache, Inf)

    # Setup Linsolve
    Δzw = similar(z, 2N)
    A_ = __make_ipm_linsolve_operator(M, z, w, Δzw)
    b_ = similar(z, 2N)
    lincache = init(LinearProblem(A_, b_; u0=Δzw), alg.linsolve; kwargs...)

    iter = 1
    while τ > η && ρ > η && iter ≤ maxiters
        lincache.A = __make_ipm_linsolve_operator(M, z, w, Δzw)
        b_[1:N] .= w .- w_cache
        b_[(N + 1):(2N)] .= σ * τ .- z .* w
        lincache.b = b_
        solve!(lincache)

        Δz, Δw = Δzw[1:length(z)], Δzw[(length(z) + 1):(2 * length(z))]

        η₁ = __feasible_steplength(z, Δz, α_cache)
        η₂ = __feasible_steplength(w, Δw, α_cache)

        z .+= η₁ .* Δz
        w .+= η₂ .* Δw

        τ = (z' * w)
        w_cache .= q
        mul!(w_cache, M, z, true, true)
        ρ = norm(w_cache, Inf)

        σ = ifelse(τ ≤ η && ρ > η,
            T(1),
            min(T(0.5), (1 - η₁)^2, (1 - η₂)^2, abs(ρ - τ) / (ρ + 10 * τ)))

        iter += 1
    end

    return LinearComplementaritySolution(z,
        [τ, ρ],
        prob,
        alg,
        iter == maxiters ? ReturnCode.MaxIters : ReturnCode.Success)
end
