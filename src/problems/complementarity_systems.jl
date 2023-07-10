abstract type AbstractComplementaritySystem{iip} end

@doc doc"""
    LinearComplementaritySystem(x0, controller, A, B, D, a, E, F, c)

LinearComplementaritySystem describes the following system:

```math
\begin{align}
    \dot{x} &= A x + B controller(x, \lambda, p) + D \lambda + a \\
    0 &\leq \lambda \perp E x + F \lambda + c \geq 0
\end{align}
```

`p` are parameters to the controller `controller`.

!!! note

    If `last(tspan)` is `Inf`, then a SteadyStateProblem is solved. This only works correctly if the `controller` is stable, else it will diverge and error out.

## References

[1] Aydinoglu, Alp, Victor M. Preciado, and Michael Posa. "Contact-aware controller design for complementarity systems." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.
"""
struct LinearComplementaritySystem{
    sstate,
    xType,
    controllerType <: Function,
    λType,
    tType,
    pT,
    AT,
    BT,
    DT,
    aT,
    ET,
    FT,
    cT,
} <: AbstractComplementaritySystem{false}
    x0::xType
    controller::controllerType
    λ0::λType
    tspan::tType
    p::pT
    A::AT
    B::BT
    D::DT
    a::aT
    E::ET
    F::FT
    c::cT
end

const LCS = LinearComplementaritySystem

function LCS(x0::AbstractVecOrMat, controller, tspan, p, A, B, D, a, E, F, c)
    sstate = isinf(last(tspan))
    λ = similar(x0, ndims(x0) == 1 ? (size(F, 2),) : similar(x0, size(F, 2), size(x0, 2)))
    CRC.@ignore_derivatives fill!(λ, zero(eltype(x0)))
    return LCS{
        sstate,
        typeof(x0),
        typeof(controller),
        typeof(λ),
        typeof(tspan),
        typeof(p),
        typeof(A),
        typeof(B),
        typeof(D),
        typeof(a),
        typeof(E),
        typeof(F),
        typeof(c),
    }(x0,
        controller,
        λ,
        tspan,
        p,
        A,
        B,
        D,
        a,
        E,
        F,
        c)
end
