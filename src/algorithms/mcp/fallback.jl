# @views function solve(prob::LinearComplementarityProblem{iip, true},
#     solver::AbstractComplementarityAlgorithm;
#     kwargs...) where {iip}
#     @warn "Solver: $(nameof(typeof(solver))) doesn't support batched problems. Falling \
#            back to iterating over the batch dimension. This is going to be slow!!" maxlog=1
#     # Do the first solve to get the types
#     sol_first = solve(LinearComplementarityProblem{iip}(prob.M[:, :, 1],
#             prob.q[:, 1],
#             prob.u0[:, 1]),
#         solver;
#         kwargs...)

#     us = similar(sol_first.u, length(sol_first.u), size(prob.u0, 2))
#     # Residual might be nothing
#     residual = sol_first.residual === nothing ? nothing :
#                similar(sol_first.residual, length(sol_first.residual), size(prob.u0, 2))
#     return_codes = Vector{ReturnCode.T}(undef, size(prob.u0, 2))
#     return_codes[1] = sol_first.retcode

#     Threads.@threads for i in 1:size(prob.u0, 2)
#         sol = solve(LinearComplementarityProblem{iip}(prob.M[:, :, i],
#                 prob.q[:, i],
#                 prob.u0[:, i]),
#             solver;
#             kwargs...)
#         us[:, i] .= sol.u
#         residual !== nothing && (residual[:, i] .= sol.residual)
#         return_codes[i] = sol.retcode
#     end

#     return LinearComplementaritySolution(us, residual, prob, solver, maximum(return_codes))
# end
