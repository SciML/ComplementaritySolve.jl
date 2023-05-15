@kwdef struct LinearComplementarityAdjoint{L}
    linsolve::L = nothing
end

_Jq(z) = Diagonal((x -> isapprox(x, 0; rtol=1e-5, atol=1e-5) ? x : one(x)).(z))

# TODO: Implement the method to account for ∂w
@views function ∇linear_complementarity_problem!(alg::LinearComplementarityAdjoint, ∂z, z,
                                                 ::Nothing, w, ∂M, M, ∂q, q)
    u₋, v₋ = w, z
    den = @. inv(√(u₋^2 + v₋^2))
    ∂ϕ₋∂u₋ = Diagonal(@. 1 - u₋ * den)
    ∂ϕ₋∂v₋ = Diagonal(@. 1 - v₋ * den)

    A = ∂ϕ₋∂u₋ * M + ∂ϕ₋∂v₋
    B = -hcat(reshape(reshape(z, 1, 1, :) .* repeat(∂ϕ₋∂u₋; outer=(1, 1, length(z))),
                      length(z), length(M)), _Jq(z))

    # To compute ∂z∂θ we need to do multiple linear solves
    # TODO: There should be a more efficient way to solve this
    ∂z∂θ = similar(z, length(z), length(M) + length(q))
    foreach(axes(∂z∂θ, 2)) do col
        # NOTE: using the set_A, set_b API is giving incorrect answers
        sol = solve(LinearProblem(A, B[:, col]), alg.linsolve)
        ∂z∂θ[:, col] .= sol.u
        return nothing
    end

    vec(∂M) .+= vec(reshape(∂z, 1, :) * ∂z∂θ[:, 1:length(M)])
    vec(∂q) .+= vec(reshape(∂z, 1, :) * ∂z∂θ[:, (length(M) + 1):end])

    return ∂M, ∂q
end

function CRC.rrule(::typeof(solve), prob::LinearComplementarityProblem, alg, args...;
                   sensealg=LinearComplementarityAdjoint(), kwargs...)
    sol = solve(prob, alg, args...; kwargs...)

    function ∇lcpsolve(Δ)
        ∂M, ∂q = zero(prob.M), zero(prob.q)
        ∇linear_complementarity_problem!(sensealg, Δ.z, sol.z, nothing, sol.w, ∂M, prob.M,
                                         ∂q, prob.q)
        ∂prob = (; M=∂M, q=∂q, u0=NoTangent())
        return (NoTangent(), ∂prob, NoTangent(), ntuple(_ -> NoTangent(), length(args))...)
    end

    return sol, ∇lcpsolve
end
