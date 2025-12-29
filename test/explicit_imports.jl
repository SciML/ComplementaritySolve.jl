using ExplicitImports
using ComplementaritySolve
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(ComplementaritySolve) === nothing
    @test check_no_stale_explicit_imports(ComplementaritySolve) === nothing
end
