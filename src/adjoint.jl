@kwdef struct LinearComplementarityAdjoint{L}
    linsolve::L = nothing
end

_Jq(z) = Diagonal((x -> isapprox(x, 0; rtol=1e-5, atol=1e-5) ? x : one(x)).(z))

@views function ∇linear_complementarity_problem!(alg::LinearComplementarityAdjoint, ∂z, z,
                                                 ∂w, w, ∂M, M, ∂q, q)
    ∂w === nothing && ∂z === nothing && return (∂0, ∂0)

    if ∂w !== nothing
        sum!(∂q, ∂w)
        mul!(∂M, ∂w, z')
        if ∂z === nothing
            ∂z = M' * ∂w
        elseif ArrayInterfaceCore.can_setindex(∂z)
            ∂z .+= M' * ∂w
        else
            ∂z = ∂z .+ M' * ∂w
        end
    end

    u₋, v₋ = w, z
    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = Diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = Diagonal(@. 1 - v₋ * den)

    A = ∂ϕ₋∂u₋ * M + ∂ϕ₋∂v₋
    B = -hcat(reshape(reshape(z, 1, 1, :) .* repeat(∂ϕ₋∂u₋; outer=(1, 1, length(z))),
                      length(z), length(M)), _Jq(z))

    λ = solve(LinearProblem(A, ∂z), alg.linsolve).u
    ∂Mq = λ' * B

    vec(∂M) .+= vec(∂Mq[1, 1:length(M)])
    vec(∂q) .+= vec(∂Mq[1, (length(M) + 1):end])

    return ∂M, ∂q
end

function CRC.rrule(::typeof(solve), prob::LinearComplementarityProblem, alg, args...;
                   sensealg=LinearComplementarityAdjoint(), kwargs...)
    sol = solve(prob, alg, args...; kwargs...)

    function ∇lcpsolve(Δ)
        ∂M, ∂q = zero(prob.M), zero(prob.q)
        ∇linear_complementarity_problem!(sensealg, __nothingify(Δ.z), sol.z,
                                         __nothingify(Δ.w), sol.w, ∂M, prob.M, ∂q, prob.q)
        ∂prob = (; M=∂M, q=∂q, u0=∂∅)
        return (∂∅, ∂prob, ∂∅, ntuple(_ -> ∂∅, length(args))...)
    end

    return sol, ∇lcpsolve
end
