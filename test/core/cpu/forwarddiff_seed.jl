using ComplementaritySolve, ForwardDiff, JLArrays, Test

# ForwardDiff 1.x seeds dual arrays with scalar indexing, which errors on GPU
# arrays. ComplementaritySolve pirates broadcast-based `seed!` methods for
# `AbstractGPUArray`; JLArrays emulates GPU array semantics on the CPU so the
# overloads can be exercised without a GPU.
JLArrays.allowscalar(false)

@testset "ForwardDiff seeding on GPU arrays" begin
    f(x) = x .^ 2 .+ 2 .* x

    @testset "vector mode (length $(n))" for n in (4, 8)
        x = collect(Float64, 1:n)
        @test Array(ForwardDiff.jacobian(f, JLArray(x))) == ForwardDiff.jacobian(f, x)
    end

    # lengths above the default chunk size exercise the chunked `seed!` methods
    @testset "chunk mode (length $(n))" for n in (16, 20, 27)
        x = collect(Float64, 1:n)
        @test Array(ForwardDiff.jacobian(f, JLArray(x))) == ForwardDiff.jacobian(f, x)
    end
end
