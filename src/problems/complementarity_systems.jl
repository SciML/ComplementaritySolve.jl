"""
    AbstractComplementaritySystem{iip}

Developer interface for complementarity system containers.

Complementarity systems couple continuous dynamics with complementarity variables.
Generic system algorithms are expected to call `solve(prob, alg; kwargs...)` and use
the concrete problem fields documented by the system type.

# Interface

- `prob.x0` must contain the initial continuous state.
- `prob.λ0` must contain the initial complementarity variable.
- `prob.tspan` must contain the integration interval.
- `prob.p` must contain parameters passed to the controller.
- `prob.controller` must be callable as `controller(x, λ, p, t)`.
"""
abstract type AbstractComplementaritySystem{iip} end

"""
    LinearComplementaritySystem(x0, controller, λ0, tspan, p, A, B, D, a, E, F, c)
    LCS(x0, controller, tspan, p, A, B, D, a, E, F, c)

LinearComplementaritySystem describes the following system:

```math
\\begin{align}
    \\dot{x} &= A x + B controller(x, \\lambda, p) + D \\lambda + a \\\\
    0 &\\leq \\lambda \\perp E x + F \\lambda + c \\geq 0
\\end{align}
```

`p` are parameters to the controller `controller`.

# Arguments

- `x0`: Initial state.
- `controller`: Control law called as `controller(x, λ, p, t)`.
- `tspan`: Time span. If `last(tspan)` is infinite, the system is solved as a
    steady-state problem.
- `p`: Parameters passed to `controller`.
- `A`, `B`, `D`, `a`: State dynamics matrices and affine offset.
- `E`, `F`, `c`: Complementarity matrices and affine offset.

# Fields

- `x0`: Initial state.
- `controller`: Control law.
- `λ0`: Initial complementarity variable.
- `tspan`: Time span.
- `p`: Controller parameters.
- `A`, `B`, `D`, `a`, `E`, `F`, `c`: System matrices and offsets.

!!! note

    If `last(tspan)` is `Inf`, then a SteadyStateProblem is solved. This only
    works correctly if the `controller` is stable, else it will diverge and error
    out.

## References

[1] Aydinoglu, Alp, Victor M. Preciado, and Michael Posa. "Contact-aware
controller design for complementarity systems." 2020 IEEE International Conference
on Robotics and Automation (ICRA). IEEE, 2020.
"""
@concrete struct LinearComplementaritySystem{sstate, controllerType <: Function} <:
    AbstractComplementaritySystem{false}
    x0
    controller::controllerType
    λ0
    tspan
    p
    A
    B
    D
    a
    E
    F
    c
end

"""
    LCS

Alias for [`LinearComplementaritySystem`](@ref).
"""
const LCS = LinearComplementaritySystem

function LCS(x0::AbstractVecOrMat, controller, tspan, p, A, B, D, a, E, F, c)
    sstate = isinf(last(tspan))
    λ = similar(x0, ndims(x0) == 1 ? (size(F, 2),) : similar(x0, size(F, 2), size(x0, 2)))
    CRC.@ignore_derivatives fill!(λ, zero(eltype(x0)))
    return LCS{sstate}(x0, controller, λ, tspan, p, A, B, D, a, E, F, c)
end
