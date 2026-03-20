@inline ϕ₊(u, v) = u + v + √(u^2 + v^2)
@inline ϕ₋(u, v) = u + v - √(u^2 + v^2)
@inline ϕ₊(fx, x, bound) = isfinite(bound) ? ϕ₊(fx, x - bound) : fx
@inline ϕ₋(fx, x, bound) = isfinite(bound) ? ϕ₋(fx, x - bound) : fx

# FIXME(@avik-pal): We are unnecessarily repeating computations here
Jϕ₊u₊(u₊, v₊, bound) = oftype(u₊, 1) + u₊ / √(u₊^2 + v₊^2)
Jϕ₊v₊(u₊, v₊, bound) = isfinite(bound) ? (one(u₊) + v₊ / √(u₊^2 + v₊^2)) : zero(u₊)
@inline function Jϕ₊(fx, x, bound)
    v₊ = x .- bound
    return __diagonal(Jϕ₊u₊.(fx, v₊, bound)), __diagonal(Jϕ₊v₊.(fx, v₊, bound))
end

Jϕ₋u₋(u₋, v₋, bound) = oftype(u₋, 1) - u₋ / √(u₋^2 + v₋^2)
Jϕ₋v₋(u₋, v₋, bound) = isfinite(bound) ? (one(u₋) - v₋ / √(u₋^2 + v₋^2)) : zero(u₋)
@inline function Jϕ₋(fx, x, bound)
    v₋ = x .- bound
    return __diagonal(Jϕ₋u₋.(fx, v₋, bound)), __diagonal(Jϕ₋v₋.(fx, v₋, bound))
end

@inline minmax_transform(fx, x, lb, ub) = min(max(fx, x - ub), x - lb)
@inline smooth_transform(fx, x, lb, ub) = ϕ₋(ϕ₊(fx, x, ub), x, lb)

# Assumes lb = 0 and ub = +∞
@inline minmax_transform(fx, x) = min(fx, x)
@inline smooth_transform(fx, x) = ϕ₋(fx, x)

# Make ZeroTangent and NoTangent to nothing
__nothingify(::ZeroTangent) = nothing
__nothingify(::NoTangent) = nothing
__nothingify(x) = x

__notangent(::Nothing) = true
__notangent(::ZeroTangent) = true
__notangent(::NoTangent) = true
__notangent(::Any) = false

__unfillarray(x::AbstractFill) = collect(x)
__unfillarray(x) = x

function batched_matvec(A::AA3, x::AM)
    y = similar(x, promote_type(eltype(x), eltype(A)), size(A, 1), size(x, 2))
    return batched_matvec!(y, A, x, true, false)
end
function batched_matvec!(y::AM, A::AA3, x::AM, α, β)
    y_ = reshape(y, size(y, 1), 1, size(y, 2))
    x_ = reshape(x, size(x, 1), 1, size(x, 2))
    batched_mul!(y_, A, x_, α, β)
    return dropdims(y_; dims = 2)
end

## Matmul with proper dispatches
matmul(A::AM, x::Union{AV, AM}) = A * x
matmul(A::AA3, x::AM) = batched_matvec(A, x)
matmul(A::AA3, x::AA3) = batched_mul(A, x)
matmul!(y::Union{AV, AM}, A::AM, x::Union{AV, AM}, α, β) = mul!(y, A, x, α, β)
matmul!(y::AM, A::AA3, x::AM, α, β) = batched_matvec!(y, A, x, α, β)
matmul!(y::AA3, A::AA3, x::AA3, α, β) = batched_mul!(y, A, x, α, β)

# Diagonal Utilities
## FIXME: This needs to be optimized to not use an insane amount of
## useless meemory, but for now this would work.
function __diagonal(x::AM)
    L, N = size(x)
    y = similar(x, (L, L, N))
    fill!(y, eltype(x)(0))
    ind = diagind(selectdim(y, 3, 1))
    for (i, x_) in enumerate(eachcol(x))
        selectdim(y, 3, i)[ind] .= x_
    end
    return y
end
__diagonal(x::AV) = Diagonal(x)

function __make_block_diagonal_operator(x::AA3)
    L, M, N = size(x)  # L == M
    @views function matvec(v::AV, u::AV, _u, p, t)
        @batch per = core for i in 1:N
            mul!(v[((i - 1) * L + 1):(i * L)], x[:, :, i], u[((i - 1) * L + 1):(i * L)])
        end
        return v
    end
    proto = similar(x, N * L)
    return FunctionOperator(matvec, proto, similar(proto); isinplace = true)
end

function __check_correct_batching(args...)
    return __check_correct_batching(map(x -> size(x, ndims(x)), args)...)
end
function __check_correct_batching(args::Int...)
    batch_size = maximum(args)
    foreach(args) do arg
        return arg != 1 && arg != batch_size && throw(ArgumentError("Incorrect batching"))
    end
    return batch_size
end

# Pseudo Inverse
I➕x⁻¹(x::AbstractMatrix) = pinv(I + x)
@views function I➕x⁻¹(x::AbstractArray{T, 3}) where {T}
    y = similar(x)
    for i in axes(x, 3)
        y[:, :, i] .= pinv(I + x[:, :, i])
    end
    return y
end

I➖x(x::AbstractMatrix) = I - x
@views function I➖x(x::AbstractArray{T, 3}) where {T}
    y = similar(x)
    for i in axes(x, 3)
        y[:, :, i] .= I - x[:, :, i]
    end
    return y
end
