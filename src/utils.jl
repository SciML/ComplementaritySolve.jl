@inline ϕ₊(u, v) = u + v + √(u^2 + v^2)
@inline ϕ₋(u, v) = u + v - √(u^2 + v^2)
@inline ϕ₊(fx, x, bound) = isfinite(bound) ? ϕ₊(fx, x - bound) : fx
@inline ϕ₋(fx, x, bound) = isfinite(bound) ? ϕ₋(fx, x - bound) : fx

@inline minmax_transform(fx, x, lb, ub) = min(max(fx, x - ub), x - lb)
@inline smooth_transform(fx, x, lb, ub) = ϕ₋(ϕ₊(fx, x, ub), x, lb)

# Assumes lb = 0 and ub = +∞
@inline minmax_transform(fx, x) = min(fx, x)
@inline smooth_transform(fx, x) = ϕ₋(fx, x)
