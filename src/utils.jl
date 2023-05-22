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
