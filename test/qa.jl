using SciMLTesting, ComplementaritySolve, Test

# TEMPORARY DIAGNOSTIC — revert before review.
# Aqua's persistent-tasks probe runs `Pkg.precompile(; io = devnull)`, so when its
# wrapper subprocess dies the cause is discarded and every failure mode looks
# identical. Reproduce the probe here with stderr passed through, and report the
# subprocess exit code and termination signal, to find out what actually happens
# on CI. Not reproducible locally: 0/25 cold recompiles fail, and a full cold
# precompile capped at 4 CPUs / 16 GB succeeds in 214s.
using Pkg
let
    pkgpath = pkgdir(ComplementaritySolve)
    wrapperdir = tempname()
    wrappername, _ = only(Pkg.generate(wrapperdir))
    prev_project = Base.active_project()
    isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(false)
    try
        Pkg.activate(wrapperdir)
        Pkg.develop(Pkg.PackageSpec(path = pkgpath))
        statusfile = joinpath(wrapperdir, "done.log")
        open(joinpath(wrapperdir, "src", wrappername * ".jl"), "w") do io
            println(
                io, """
                module $wrappername
                using ComplementaritySolve
                open("$(escape_string(statusfile))", "w") do io
                    println(io, "done"); flush(io)
                end
                end
                """
            )
        end
        @info "DIAGNOSTIC: starting wrapper precompile (stderr passed through)"
        cmd = `$(Base.julia_cmd()) --project=$wrapperdir -e 'push!(LOAD_PATH, "@stdlib"); using Pkg; Pkg.precompile()'`
        t0 = time()
        proc = run(cmd, stdin, stdout, stderr; wait = false)
        while !isfile(statusfile) && process_running(proc)
            sleep(0.5)
        end
        ok = isfile(statusfile)
        wait(proc)
        @info "DIAGNOSTIC: wrapper precompile finished" done_log_written = ok exitcode = proc.exitcode termsignal = proc.termsignal elapsed_s = round(time() - t0, digits = 1)
    finally
        isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(true)
        Pkg.activate(prev_project)
    end
end

run_qa(
    ComplementaritySolve;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :MCP_MajorIterationLimit, :MCP_MinorIterationLimit,
                :MCP_NoProgress, :MCP_Solved, :MCP_TimeLimit, :solve_mcp,
            ),
        ),
    ),
)
