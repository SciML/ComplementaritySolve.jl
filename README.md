# ComplementaritySolve.jl

A package for solving complementarity problems in Julia with scalable gradients compatible
with ChainRules.

## Installation

```julia
import Pkg
Pkg.add("https://github.com:avik-pal/ComplementaritySolve.jl.git")
```

## Implemented Problems & Algorithms

### LCP Solvers

| Solver                     |  Native Batching   |        CPU         |      CUDA[^1]      | Details         |
| -------------------------- | :----------------: | :----------------: | :----------------: | --------------- |
| NonlinearReformulation     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                 |
| RPGS                       |        :x:         | :heavy_check_mark: |        :x:         |                 |
| PGS                        |        :x:         | :heavy_check_mark: |        :x:         |                 |
| PSOR                       |        :x:         | :heavy_check_mark: |        :x:         |                 |
| BokhovenIterativeAlgorithm | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Assumes PSD `M` |
| InteriorPointMethod        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Assumes PSD `M` |

Solvers that don't natively support batching, use threads to solve multiple problems in parallel.

### MCP Solvers

| Solver                 |        CPU         |    CUDA[^1]    | Details                                                                                                                                                              |
| ---------------------- | :----------------: | :------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NonlinearReformulation | :heavy_check_mark: | :question:[^2] |                                                                                                                                                                      |
| PATHSolver             | :heavy_check_mark: |      :x:       | * Provides an uniform API to access [path.c](https://pages.cs.wisc.edu/~ferris/path.html) <br/> * Only `Float64` is supported, all inputs will be cast to `Float64`. |

All `LCP`s, `MLCP`s, and `NCP`s can be converted to `MCP`s, and these solvers can be used directly.

### Adjoint Methods

| Method                       | Problem Type |  Native Batching   |        CPU         |    CUDA[^1]    | Details |
| ---------------------------- | ------------ | :----------------: | :----------------: | :------------: | ------- |
| LinearComplementarityAdjoint | LCP          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |         |
| MixedComplementarityAdjoint  | MCP          |                    | :heavy_check_mark: | :question:[^2] |         |

[^1]: Solvers internally using `NonlinearSolve.jl` need to use a CUDA compatible solver
(like `SimpleNewtonRaphson(; batched=true)`).
[^2]: Untested.

## Usage

More details are WIP. Examples can be found in `test` directory.

## Current Unique Features of ComplementaritySolve.jl:

* Supports batched problems for LCPs.
* Can use any arbitrary forward solver. If Siconos is faster, we can use that solver, and we will still be able to give gradients to the end user.
* Most solvers should work on GPUs OOTB.
* Scalable Adjoint computation using Linear Solve trick rather than computing explicit Jacobians like ParametricMCPs.jl.
  * If solver thinks problem is small enough, we still construct the Jacobian.
