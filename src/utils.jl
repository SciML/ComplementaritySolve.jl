@inline ϕ₊(u, v) = u + v + √(u^2 + v^2)
@inline ϕ₋(u, v) = u + v - √(u^2 + v^2)
@inline ϕ₊(fx, x, bound) = isfinite(bound) ? ϕ₊(fx, x - bound) : fx
@inline ϕ₋(fx, x, bound) = isfinite(bound) ? ϕ₋(fx, x - bound) : fx

# FIXME(@avik-pal): We are unnecessarily repeating computations here
Jϕ₊u₊(u₊, v₊, bound) = oftype(u₊, 1) + u₊ / √(u₊^2 + v₊^2)
Jϕ₊v₊(u₊, v₊, bound) = isfinite(bound) ? (one(u₊) + v₊ / √(u₊^2 + v₊^2)) : zero(u₊)
@inline function Jϕ₊(fx, x, bound)
    v₊ = x .- bound
    return Diagonal(Jϕ₊u₊.(fx, v₊, bound)), Diagonal(Jϕ₊v₊.(fx, v₊, bound))
end

Jϕ₋u₋(u₋, v₋, bound) = oftype(u₋, 1) - u₋ / √(u₋^2 + v₋^2)
Jϕ₋v₋(u₋, v₋, bound) = isfinite(bound) ? (one(u₋) - v₋ / √(u₋^2 + v₋^2)) : zero(u₋)
@inline function Jϕ₋(fx, x, bound)
    v₋ = x .- bound
    return Diagonal(Jϕ₋u₋.(fx, v₋, bound)), Diagonal(Jϕ₋v₋.(fx, v₋, bound))
end

@inline minmax_transform(fx, x, lb, ub) = min(max(fx, x - ub), x - lb)
@inline smooth_transform(fx, x, lb, ub) = ϕ₋(ϕ₊(fx, x, ub), x, lb)

# Assumes lb = 0 and ub = +∞
@inline minmax_transform(fx, x) = min(fx, x)
@inline smooth_transform(fx, x) = ϕ₋(fx, x)

# Make ZeroTangent and NoTangent to nothing
__nothingify(::ZeroTangent) = nothing
__nothingify(::NoTangent) = nothing
__nothingify(x) = x

__notangent(::Nothing) = true
__notangent(::ZeroTangent) = true
__notangent(::NoTangent) = true
__notangent(::Any) = false

__unfillarray(x::FillArrays.AbstractFill) = collect(x)
__unfillarray(x) = x

# Diagonal Utilities
## FIXME: This needs to be optimized to not use an insane amount of
## useless meemory, but for now this would work.
function __diagonal(x::AbstractMatrix)
    L, N = size(x)
    y = similar(x, (L, L, N))
    fill!(y, eltype(x)(0))
    ind = diagind(selectdim(y, 3, 1))
    for (i, x_) in enumerate(eachcol(x))
        selectdim(y, 3, i)[ind] .= x_
    end
    return y
end
__diagonal(x) = Diagonal(x)

function __make_square(x::AbstractMatrix)
    L, N = size(x)
    L == N && return x
    if L > N
        y = similar(x, (L, L - N))
        fill!(y, eltype(x)(0))
        return hcat(x, y)
    else
        y = similar(x, (N - L, N))
        fill!(y, eltype(x)(0))
        return vcat(x, y)
    end
end

## FIXME: Don't Allocate the entire matrix
function __make_banded_diagonal_matrix(x::AbstractArray{<:Number, 3})
    L, K, N = size(x)
    @assert L == K
    y = similar(x, (L * N, L * N))
    fill!(y, eltype(x)(0))
    for (i, x_) in enumerate(eachslice(x; dims=3))
        y[(i - 1) * L + 1:i * L, (i - 1) * L + 1:i * L] .= x_
    end
    return y
end
