"""
    AbstractComplementarityProblem{iip}

Developer interface for complementarity problem containers.

The `iip` type parameter records whether the residual function follows the in-place
SciML function convention. Generic code queries this trait with
`SciMLBase.isinplace(prob)`.

# Interface

- `prob.u0` must contain the initial guess or state used by `solve`.
- Nonlinear problem subtypes must provide `prob.p` for parameters.
- Subtypes must dispatch to a supported `__solve` method through
    `solve(prob, alg; kwargs...)`.
- If `iip == true`, residual functions must support `f(residual, u, p)`.
- If `iip == false`, residual functions must support `f(u, p)`.

This is a developer extension point. External users should construct one of the
concrete problem types exported by ComplementaritySolve.
"""
abstract type AbstractComplementarityProblem{iip} end

"""
    AbstractLinearComplementarityProblem{iip, batched}

Abstract interface for linear complementarity problems.

Linear complementarity problems represent
`0 <= u ⟂ M * u + q >= 0`.

# Interface

- `prob.M` must contain the matrix or linear operator.
- `prob.q` must contain the affine offset.
- `prob.u0` must contain the initial guess.
- The `batched` type parameter must be `true` when columns of `u0`/`q` and slices of
    `M` represent independent problems.
- `isbatched(prob)` returns the `batched` trait.
- `solve(prob, alg; u0, M, q, kwargs...)` may override `prob.u0`, `prob.M`, and
    `prob.q` for a single solve.
"""
abstract type AbstractLinearComplementarityProblem{iip, batched} <:
AbstractComplementarityProblem{iip} end

"""
    AbstractNonlinearComplementarityProblem{iip} <: AbstractComplementarityProblem{iip}

Abstract interface for nonlinear and mixed complementarity problems.

# Interface

- `prob.f` must contain the residual function.
- `prob.u0` must contain the initial state.
- `prob.p` must contain the parameter object passed to `prob.f`.
- `solve(prob, alg; u0, p, kwargs...)` may override `prob.u0` and `prob.p` for a
    single solve.
- The residual must follow the `iip` convention used by `SciMLBase.isinplace`.
"""
abstract type AbstractNonlinearComplementarityProblem{iip} <:
AbstractComplementarityProblem{iip} end

SciMLBase.isinplace(::AbstractComplementarityProblem{iip}) where {iip} = iip
isbatched(::AbstractLinearComplementarityProblem{I, B}) where {I, B} = B

"""
    LinearComplementarityProblem{iip}(M, q, u0 = nothing)
    LinearComplementarityProblem(M, q, u0 = nothing)

Define a linear complementarity problem (LCP)
`0 <= u ⟂ M * u + q >= 0`.

# Arguments

- `M`: Linear operator or matrix. A three-dimensional array encodes batched LCPs.
- `q`: Offset vector or matrix. For batched problems, columns correspond to batches.
- `u0`: Optional initial guess. If omitted, a zero-valued array matching `q` is used.

# Type Parameters

- `iip`: Whether generated residual functions are in-place. The default constructor
    uses `true`.
- `batched`: Inferred from `M`, `q`, and `u0`; `true` when the problem stores a batch
    of independent LCPs.

# Fields

- `M`: Linear operator data.
- `q`: Offset data.
- `u0`: Initial guess used by algorithms.

# Examples

```julia
using ComplementaritySolve

M = [2.0 -1.0; -1.0 2.0]
q = [-1.0, -1.0]
prob = LinearComplementarityProblem(M, q)
sol = solve(prob, PGS())
```
"""
@concrete struct LinearComplementarityProblem{iip, batched} <:
    AbstractLinearComplementarityProblem{iip, batched}
    M
    q
    u0
end

function LinearComplementarityProblem{iip}(M, q, u0 = nothing) where {iip}
    # By default, set iip to true since that is faster
    if u0 !== nothing && ndims(u0) == 2 && ndims(M) == 2 && ndims(q) == 1
        # If u0 is batched while problem is not, then reshape the problem
        M = reshape(M, size(M)..., 1)
        q = reshape(q, length(q), 1)
    end

    if ndims(M) == 3 && ndims(q) == 1
        q = reshape(q, length(q), 1)
    elseif ndims(M) == 2 && ndims(q) == 2
        M = reshape(M, size(M)..., 1)
    end

    batched = ndims(M) == 3
    batched && (batch_size = __check_correct_batching(M, q))

    if u0 === nothing
        u0 = similar(q, batched ? (size(q, 1), batch_size) : size(q))
        fill!(u0, 0)
    elseif batched
        @assert ndims(u0) == 2
        batch_size > 1 && @assert size(u0, 2) == batch_size
    end

    return LinearComplementarityProblem{iip, batched}(M, q, u0)
end

LinearComplementarityProblem(args...) = LinearComplementarityProblem{true}(args...)

for iip in (true, false)
    @eval function CRC.rrule(
            ::Type{LinearComplementarityProblem{$iip}}, M, q, args...;
            kwargs...
        )
        prob = LinearComplementarityProblem{$iip}(M, q, args...; kwargs...)
        function ∇LinearComplementarityProblem(Δ)
            if __notangent(Δ)
                ∂M = ∂∅
                ∂q = ∂∅
            else
                if isbatched(prob)
                    if ndims(M) != ndims(Δ.M)
                        ∂M = dropdims(sum(Δ.M; dims = ndims(Δ.M)); dims = ndims(Δ.M))
                    end
                    if ndims(q) != ndims(Δ.q)
                        ∂q = dropdims(sum(Δ.q; dims = ndims(Δ.q)); dims = ndims(Δ.q))
                    end
                end
                @isdefined(∂M) || (∂M = Δ.M)
                @isdefined(∂q) || (∂q = Δ.q)
            end
            return ∂∅, ∂M, ∂q, ∂0
        end
        return prob, ∇LinearComplementarityProblem
    end
end


"""
    LCP

Alias for [`LinearComplementarityProblem`](@ref).
"""
const LCP = LinearComplementarityProblem

function (prob::LCP{iip, batched})(M = prob.M, q = prob.q) where {iip, batched}
    ff = if iip
        function f!(out, u, θ)
            M = reshape(view(θ, 1:length(M)), size(M))
            q = reshape(view(θ, (length(M) + 1):length(θ)), size(q))
            out .= q
            matmul!(out, M, u, true, true)
            return out
        end
    else
        function f(u, θ)
            M = reshape(view(θ, 1:length(M)), size(M))
            q = reshape(view(θ, (length(M) + 1):length(θ)), size(q))
            return matmul(M, u) .+ q
        end
    end

    return ff, vcat(vec(M), vec(q))
end

"""
    MixedLinearComplementarityProblem{iip, batched}(M, q, u0, lb, ub)

Define a mixed linear complementarity problem with finite or infinite bounds.

# Arguments

- `M`: Linear operator or matrix.
- `q`: Offset vector or matrix.
- `u0`: Initial state.
- `lb`: Lower bounds for each complementarity variable.
- `ub`: Upper bounds for each complementarity variable.

# Fields

- `M`: Linear operator data.
- `q`: Offset data.
- `u0`: Initial guess.
- `lb`: Lower bounds.
- `ub`: Upper bounds.
"""
@concrete struct MixedLinearComplementarityProblem{iip, batched} <:
    AbstractLinearComplementarityProblem{iip, batched}
    M
    q
    u0
    lb
    ub
end


"""
    MLCP

Alias for [`MixedLinearComplementarityProblem`](@ref).
"""
const MLCP = MixedLinearComplementarityProblem

function MLCP(prob::LCP{iip, batched}) where {iip, batched}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MLCP{iip, batched}(prob.M, prob.q, prob.u0, lb, ub)
end

"""
    NonlinearComplementarityProblem{iip}(f, u0, p)

Define a nonlinear complementarity problem (NCP)
`0 <= u ⟂ f(u, p) >= 0`.

# Arguments

- `f`: Residual function. For `iip == true`, it must support `f(residual, u, p)`.
    For `iip == false`, it must support `f(u, p)`.
- `u0`: Initial state.
- `p`: Parameters passed to `f`.

# Fields

- `f`: Residual function.
- `u0`: Initial state.
- `p`: Parameters.
"""
@concrete struct NonlinearComplementarityProblem{iip, F <: Function} <:
    AbstractNonlinearComplementarityProblem{iip}
    f::F
    u0
    p
end


"""
    NCP

Alias for [`NonlinearComplementarityProblem`](@ref).
"""
const NCP = NonlinearComplementarityProblem

function NCP(prob::LCP{iip}) where {iip}
    f, θ = prob()
    return NCP{iip}(f, prob.u0, θ)
end

"""
    MixedComplementarityProblem{iip}(f, u0, lb, ub, p)
    MCP(f, u0, lb, ub, p)

Define a mixed complementarity problem with lower and upper bounds.

# Arguments

- `f`: Residual function. In-place functions use `f(residual, u, p)`; out-of-place
    functions use `f(u, p)`.
- `u0`: Initial state.
- `lb`: Lower bounds.
- `ub`: Upper bounds.
- `p`: Parameters passed to `f`.

# Fields

- `f`: Residual function.
- `u0`: Initial state.
- `lb`: Lower bounds.
- `ub`: Upper bounds.
- `p`: Parameters.
"""
@concrete struct MixedComplementarityProblem{iip, F <: Function} <:
    AbstractNonlinearComplementarityProblem{iip}
    f::F
    u0
    lb
    ub
    p
end


"""
    MCP

Alias for [`MixedComplementarityProblem`](@ref). `MCP(prob)` converts an `LCP` or
`NCP` to the corresponding mixed problem representation.
"""
const MCP = MixedComplementarityProblem

MCP(prob::LCP) = MCP(NCP(prob))

function MCP(prob::NCP{iip}) where {iip}
    lb = zero(prob.u0)
    ub = similar(prob.u0)
    fill!(ub, eltype(prob.u0)(Inf))
    return MCP{iip}(prob.f, prob.u0, lb, ub, prob.p)
end

MCP(f, u0, lb, ub, p) = MCP{SciMLBase.isinplace(f, 3)}(f, u0, lb, ub, p)
