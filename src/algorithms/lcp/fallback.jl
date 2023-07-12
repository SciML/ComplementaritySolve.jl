@views function solve(prob::LinearComplementarityProblem{iip, true},
    solver::AbstractComplementarityAlgorithm;
    kwargs...) where {iip}
    @warn "Solver: $(nameof(typeof(solver))) doesn't support batched problems. Falling \
           back to iterating over the batch dimension. This is going to be slow!!" maxlog=1
    solutions = map(1:size(prob.u0, 2)) do i
        return solve(LinearComplementarityProblem(prob.M[:, :, i],
                prob.q[:, i],
                prob.u0[:, i]),
            solver;
            kwargs...)
    end

    __batch(x) = mapreduce(Base.Fix2(getproperty, x), hcat, solutions)

    return LinearComplementaritySolution(__batch(:z),
        __batch(:w),
        __batch(:resid),
        prob,
        solver)
end
