ϕ₊(u, v) = u + v + √(u^2 + v^2)
ϕ₋(u, v) = u + v - √(u^2 + v^2)
ϕ₊(fx, x, bound) = isfinite(bound) ? ϕ₊(fx, x - bound) : fx
ϕ₋(fx, x, bound) = isfinite(bound) ? ϕ₋(fx, x - bound) : fx

minmax_transform(fx, x, lb, ub) = min(max(fx, x - ub), x - lb)
smooth_transform(fx, x, lb, ub) = ϕ₋(ϕ₊(fx, x, ub), x, lb)
