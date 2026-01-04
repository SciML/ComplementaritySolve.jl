struct PATHSolverAlgorithm <: AbstractComplementarityAlgorithm end

# TODO: We might want to exploit sparsity using Symbolics.jl. Else PATH Solver won't be
#       competitive with other solvers. See ParametricMCPs.jl for an example.
function __solve(
        prob::MCP{iip}, alg::PATHSolverAlgorithm, u0, p; verbose::Bool = true,
        kwargs...
    ) where {iip}
    (; f, lb, ub) = prob

    (
        u0,
        lb,
        ub,
        p,
    ) = map((u0, lb, ub, p)) do x
        eltype(x) == Float64 && return x
        if verbose
            @warn "PATHSolver doesn't support Non Float64 ($(eltype(x))) inputs. Converted \
                   them to Float64" maxlog = 1
        end
        return Float64.(x)
    end

    fₚ = iip ? (y, x) -> f(y, x, p) : Base.Fix2(f, p)
    n = length(u0)

    function F!(n, z, y)
        iip ? fₚ(y, z) : (y .= fₚ(z))
        return Cint(0)
    end

    function J!(n, nnz, z, col, len, row, data)
        if !iip
            J = (n ≤ 100 ? ForwardDiff.jacobian : Zygote.jacobian)(fₚ, z)
        else
            J = ForwardDiff.jacobian(fₚ, similar(z, n), z)
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

    status, z, info = PATHSolver.solve_mcp(F!, J!, lb, ub, u0; silent = (!verbose), kwargs...)
    return MixedComplementaritySolution(
        z, info.residual, prob, alg,
        __pathsolver_status_to_return_code(status)
    )
end

function __pathsolver_status_to_return_code(status)
    if status == PATHSolver.MCP_Solved
        return ReturnCode.Success
    elseif status == PATHSolver.MCP_NoProgress
        return ReturnCode.Terminated
    elseif status == PATHSolver.MCP_MajorIterationLimit
        return ReturnCode.MaxIters
    elseif status == PATHSolver.MCP_MinorIterationLimit
        return ReturnCode.MaxIters
    elseif status == PATHSolver.MCP_TimeLimit
        return ReturnCode.MaxTime
    else  # Errors. Currently returning a random thing
        return ReturnCode.Infeasible
    end
end
