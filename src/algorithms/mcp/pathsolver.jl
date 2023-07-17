struct PathSolverAlgorithm <: AbstractComplementarityAlgorithm end

function solve(prob::MixedComplementarityProblem{false},
    alg::PathSolverAlgorithm;
    kwargs...)
    func, u0, lb, ub, θ = prob.f, prob.u0, prob.lb, prob.ub, prob.p

    if (eltype(u0) == Float32 ||
        eltype(lb) == Float32 ||
        eltype(lb) == Float32 ||
        eltype(p) == Float32)
        @warn "PATHSolver doesn't support Float32"
        u0 = Float64.(u0)
        lb = Float64.(lb)
        ub = Float64.(ub)
        θ = Float64.(θ)
    end

    n = length(u0)

    function F(n, z, f)
        f .= func(z, θ)
        return Cint(0)
    end

    function J(n, nnz, z, col, len, row, data)
        if n <= 100
            J = ForwardDiff.jacobian(z -> func(z, θ), z)
        else
            J = Zygote.jacobian(z -> func(z, θ), z)
        end

        i = 1
        for c in 1:n
            col[c], len[c] = i, 0
            for r in 1:n
                if !iszero(J[r, c])
                    row[i], data[i] = r, J[r, c]
                    len[c] += 1
                    i += 1
                end
            end
        end
        return Cint(0)
    end

    status, z, info = PATHSolver.solve_mcp(F, J, lb, ub, u0)
    return MixedComplementaritySolution(z, info.residual, prob, alg)
end
