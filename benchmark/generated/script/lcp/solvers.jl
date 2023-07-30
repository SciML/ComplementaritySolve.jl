
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))


using CUDA, Statistics

CUDA.allowscalar(false)

iscuda(args...) = any(Base.Fix2(isa, CUDA.AnyCuArray), args)

function timer(f, args...; numtimes::Int=10)
    cuda_mode = iscuda(args...)
    f(args...) # compile
    times = zeros(numtimes)
    for i in eachindex(times)
        if cuda_mode
            times[i] = @elapsed CUDA.@sync f(args...)
        else
            times[i] = @elapsed f(args...)
        end
    end
    return minimum(times)
end


using BenchmarkTools, ComplementaritySolve, DataFrames, NonlinearSolve, PyPlot, StableRNGs


A₁ = [2.0 1; 1 2.0]
q₁ = [-5.0, 6.0]


SOLVERS = [BokhovenIterativeAlgorithm(),
    PGS(),
    RPGS(),
    InteriorPointMethod(),
    NonlinearReformulation(),
    NonlinearReformulation(:smooth, Broyden(; batched=true)),
]
times = zeros(length(SOLVERS), 2)
solvers = ["Bok.", "PGS", "RPGS", "IPM", "NLR (Newton)", "NLR (Broyden)"]

for (i, solver) in enumerate(SOLVERS)
    @info "[CPU] UNBATCHED: Benchmarking $(solvers[i])"
    prob_iip = LCP{true}(A₁, q₁, rand(StableRNG(0), 2))
    prob_oop = LCP{false}(A₁, q₁, rand(StableRNG(0), 2))
    times[i, 1] = timer(solve, prob_iip, solver)
    times[i, 2] = timer(solve, prob_oop, solver)
end


let _=plt.xkcd()
    xloc = 1:length(solvers)
    width = 0.4  # the width of the bars
    multiplier = 0
    fig, ax = subplots(layout="constrained", figsize=(14, 6))
    ax.set_yscale("log")

    for (i, group) in enumerate(["Inplace", "Out of Place"])
        offset = width * multiplier
        rects = ax.bar(xloc .+ offset, times[:, i], width, label=group)
        for (j, rect) in enumerate(rects)
            height = rect.get_height()
            ax.annotate("$(round(times[j, i] * 10^6; digits=2))μs",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom")
        end
        multiplier += 1
    end

    ax.set_ylabel("Times (s)")
    ax.set_title("[CPU] UNBATCHED: Basic LCP")
    ax.set_xticks(xloc .+ width ./ 2, solvers)
    ax.legend(ncols=3)
    fig.tight_layout()
    fig
end


SOLVERS = [BokhovenIterativeAlgorithm(Broyden(; batched=true)),
    PGS(),
    RPGS(),
    InteriorPointMethod(),
    NonlinearReformulation(:smooth, SimpleNewtonRaphson(; batched=true)),
    NonlinearReformulation(:smooth, SimpleDFSane(; batched=true)),
    NonlinearReformulation(:smooth, Broyden(; batched=true)),
]
BATCH_SIZES = 2 .^ (1:2:11)
times = zeros(length(SOLVERS), length(BATCH_SIZES), 2)
solvers = ["Bok.", "PGS", "RPGS", "IPM", "NLR (Newton)", "NLR (DFSane)", "NLR (Broyden)"]

for (i, solver) in enumerate(SOLVERS)
    @info "[CPU] BATCHED: Benchmarking $(solvers[i])"
    for (j, N) in enumerate(BATCH_SIZES)
        prob_iip = LCP{true}(A₁, q₁, rand(StableRNG(0), 2, N))
        prob_oop = LCP{false}(A₁, q₁, rand(StableRNG(0), 2, N))
        times[i, j, 2] = timer(solve, prob_oop, solver)
        if i == 5
            times[i, j, 1] = -1 # SimpleNewtonRaphson is not implemented for inplace
            continue
        end
        times[i, j, 1] = timer(solve, prob_iip, solver)
    end
end


let _=plt.xkcd()
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig, (ax1, ax2) = subplots(1, 2; layout="constrained", sharey=true, sharex=true, figsize=(16, 6))
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    fig.suptitle("[CPU] BATCHED: Basic LCP")
    ax1.set_title("In-Place Solvers")
    ax2.set_title("Out-Of-Place Solvers")

    for (j, solver) in enumerate(solvers)
        if !any(times[j, :, 1] .< 0)
            ax1.plot(BATCH_SIZES, times[j, :, 1]; label=solver, color=colors[j])
            ax1.scatter(BATCH_SIZES, times[j, :, 1]; color=colors[j])
        end
        if !any(times[j, :, 2] .< 0)
            ax2.plot(BATCH_SIZES, times[j, :, 2]; label=solver, color=colors[j])
            ax2.scatter(BATCH_SIZES, times[j, :, 2]; color=colors[j])
        end
    end

    ax1.set_ylabel("Times (s)")
    ax1.set_xlabel("Batch Size")
    ax2.set_xlabel("Batch Size")
    # ax1.legend(ncols=3)
    ax2.legend(ncols=3)
    fig.tight_layout()
    fig
end


cuA₁ = [2.0 1; 1 2.0] |> cu
cuq₁ = [-5.0, 6.0] |> cu


SOLVERS = [BokhovenIterativeAlgorithm(Broyden(; batched=true)),
    # InteriorPointMethod(),
    NonlinearReformulation(:smooth, Broyden(; batched=true)),
]
times = zeros(length(SOLVERS), 2)
solvers = ["Bok.", "NLR (Broyden)"]

for (i, solver) in enumerate(SOLVERS)
    @info "[CUDA] UNBATCHED: Benchmarking $(solvers[i])"
    prob_iip = LCP{true}(cuA₁, cuq₁, rand(StableRNG(0), 2) |> cu)
    prob_oop = LCP{false}(cuA₁, cuq₁, rand(StableRNG(0), 2) |> cu)
    times[i, 1] = timer(solve, prob_iip, solver)
    times[i, 2] = timer(solve, prob_oop, solver)
end


let _=plt.xkcd()
    xloc = 1:length(solvers)
    width = 0.4  # the width of the bars
    multiplier = 0
    fig, ax = subplots(layout="constrained", figsize=(14, 6))
    ax.set_yscale("log")

    for (i, group) in enumerate(["Inplace", "Out of Place"])
        offset = width * multiplier
        rects = ax.bar(xloc .+ offset, times[:, i], width, label=group)
        for (j, rect) in enumerate(rects)
            height = rect.get_height()
            ax.annotate("$(round(times[j, i] * 10^6; digits=2))μs",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom")
        end
        multiplier += 1
    end

    ax.set_ylabel("Times (s)")
    ax.set_title("[CUDA] UNBATCHED: Basic LCP")
    ax.set_xticks(xloc .+ width ./ 2, solvers)
    ax.legend(ncols=3)
    fig.tight_layout()
    fig
end


SOLVERS = [BokhovenIterativeAlgorithm(Broyden(; batched=true)),
    # InteriorPointMethod(),
    NonlinearReformulation(:smooth, Broyden(; batched=true)),
]
BATCH_SIZES = 2 .^ (1:2:11)
times = zeros(length(SOLVERS), length(BATCH_SIZES), 2)
solvers = ["Bok.", "NLR (Broyden)"]

for (i, solver) in enumerate(SOLVERS)
    @info "[CUDA] BATCHED: Benchmarking $(solvers[i])"
    for (j, N) in enumerate(BATCH_SIZES)
        prob_iip = LCP{true}(cuA₁, cuq₁, rand(StableRNG(0), 2, N) |> cu)
        prob_oop = LCP{false}(cuA₁, cuq₁, rand(StableRNG(0), 2, N) |> cu)
        times[i, j, 2] = timer(solve, prob_oop, solver)
        times[i, j, 1] = timer(solve, prob_iip, solver)
    end
end


let _=plt.xkcd()
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig, (ax1, ax2) = subplots(1, 2; layout="constrained", sharey=true, sharex=true, figsize=(16, 6))
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    fig.suptitle("[CUDA] BATCHED: Basic LCP")
    ax1.set_title("In-Place Solvers")
    ax2.set_title("Out-Of-Place Solvers")

    for (j, solver) in enumerate(solvers)
        if !any(times[j, :, 1] .< 0)
            ax1.plot(BATCH_SIZES, times[j, :, 1]; label=solver, color=colors[j])
            ax1.scatter(BATCH_SIZES, times[j, :, 1]; color=colors[j])
        end
        if !any(times[j, :, 2] .< 0)
            ax2.plot(BATCH_SIZES, times[j, :, 2]; label=solver, color=colors[j])
            ax2.scatter(BATCH_SIZES, times[j, :, 2]; color=colors[j])
        end
    end

    ax1.set_ylabel("Times (s)")
    ax1.set_xlabel("Batch Size")
    ax2.set_xlabel("Batch Size")
    # ax1.legend(ncols=3)
    ax2.legend(ncols=3)
    fig.tight_layout()
    fig
end


import SciMLBenchmarks
SciMLBenchmarks.bench_footer(@__DIR__, last(splitdir(@__FILE__)))

