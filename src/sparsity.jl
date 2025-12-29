"""
    compute_jacobian_sparsity(f, u0, p; iip=SciMLBase.isinplace(f, 3))

Compute the sparsity pattern of the Jacobian of `f(u, p)` (or `f(out, u, p)` for in-place)
with respect to `u` at the point `u0` with parameters `p`.

Returns a sparse matrix with the same sparsity pattern as the Jacobian.

Uses SparseConnectivityTracer.jl for fast operator-overloading based sparsity detection.

# Arguments

  - `f`: The function to analyze. Can be in-place `f(out, u, p)` or out-of-place `f(u, p)`.
  - `u0`: Initial guess vector, used to determine the input dimension.
  - `p`: Parameters passed to `f`.
  - `iip`: Whether the function is in-place (default: auto-detected).

# Returns

A sparse matrix with `true` values at non-zero positions of the Jacobian.

# Example

```julia
f(z, θ) = [2z[1] - z[2]; z[1] + z[2]]
sparsity = compute_jacobian_sparsity(f, zeros(2), nothing)
```
"""
function compute_jacobian_sparsity(f, u0, p; iip=SciMLBase.isinplace(f, 3))
    n = length(u0)

    if iip
        output = similar(u0)
        f!(y, x) = (f(y, x, p); y)
        # For in-place, we need to create a wrapper that returns output
        function f_wrapped(x)
            y = similar(x)
            f(y, x, p)
            return y
        end
        sparsity = SparseConnectivityTracer.connectivity(f_wrapped, u0)
    else
        f_oop(x) = f(x, p)
        sparsity = SparseConnectivityTracer.connectivity(f_oop, u0)
    end
    # Convert to Int indices for compatibility with UMFPACK and other solvers
    I, J, V = findnz(sparsity)
    return sparse(Int.(I), Int.(J), V, size(sparsity)...)
end

# Mark compute_jacobian_sparsity as non-differentiable since sparsity detection
# should not be traced through by AD systems
CRC.@non_differentiable compute_jacobian_sparsity(f, u0, p)

"""
    MCP(f, u0, lb, ub, p, ::Val{:auto})

Create a MixedComplementarityProblem with automatic Jacobian sparsity detection.

This constructor uses SparseConnectivityTracer.jl to detect the sparsity pattern of the
Jacobian at problem creation time, which can significantly speed up algorithms that require
Jacobian computations.

# Example

```julia
f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
u0 = zeros(4)
lb = [-Inf, -Inf, 0.0, 0.0]
ub = [Inf, Inf, Inf, Inf]
θ = [1.0, 2.0]

# Create MCP with automatic sparsity detection
prob = MCP(f, u0, lb, ub, θ, Val(:auto))
```
"""
function MCP(f, u0, lb, ub, p, ::Val{:auto})
    jac_prototype = compute_jacobian_sparsity(f, u0, p)
    return MCP(f, u0, lb, ub, p; jac_prototype=jac_prototype)
end

"""
    compute_sparse_jacobian!(J, f, u, p, colors, jac_prototype; iip=true)

Compute the sparse Jacobian of `f` at `u` with parameters `p` using matrix coloring.

This function exploits the sparsity pattern to compute the Jacobian more efficiently
by grouping columns with the same color together.

# Arguments

  - `J`: Pre-allocated sparse Jacobian matrix (same sparsity as `jac_prototype`).
  - `f`: The function whose Jacobian to compute.
  - `u`: Current point at which to evaluate the Jacobian.
  - `p`: Parameters passed to `f`.
  - `colors`: Column coloring vector from `matrix_colors(jac_prototype)`.
  - `jac_prototype`: Sparse matrix defining the sparsity pattern.
  - `iip`: Whether `f` is in-place.

# Returns

The updated sparse Jacobian `J`.
"""
function compute_sparse_jacobian!(
        J::SparseMatrixCSC, f, u, p, colors, jac_prototype; iip=true)
    if iip
        fₚ = (y, x) -> f(y, x, p)
        SparseDiffTools.forwarddiff_color_jacobian!(J, fₚ, u; colorvec=colors)
    else
        fₚ = Base.Fix2(f, p)
        SparseDiffTools.forwarddiff_color_jacobian!(J, fₚ, u; colorvec=colors)
    end
    return J
end

"""
    init_sparse_jacobian_cache(prob::MixedComplementarityProblem)

Initialize a cache for sparse Jacobian computation based on the problem's sparsity pattern.

Returns `nothing` if the problem has no sparsity pattern, otherwise returns a named tuple
with pre-allocated arrays for efficient sparse Jacobian computation.
"""
function init_sparse_jacobian_cache(prob::MixedComplementarityProblem)
    jac_prototype = prob.jac_prototype
    jac_prototype === nothing && return nothing

    # Compute matrix coloring for efficient sparse Jacobian computation
    colors = SparseDiffTools.matrix_colors(jac_prototype)

    # Pre-allocate the Jacobian matrix with the correct sparsity pattern
    J = similar(jac_prototype, eltype(prob.u0))

    return (; J, colors, jac_prototype)
end
