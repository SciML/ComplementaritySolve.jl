# ComplementaritySolve.jl

A package for solving complementarity problems in Julia with scalable gradients compatible
with ChainRules.

## Installation

```julia
] add git@github.com:avik-pal/ComplementaritySolve.jl.git
```

Add a specific version of `SimpleNonlinearSolve.jl` with
`] add https://github.com/avik-pal/SimpleNonlinearSolve.jl#ap/batch_revamp`

## Implemented Problems & Algorithms

### Complementarity Problems

* Linear Complementarity Problems (LCP)
  * Nonlinear Reformulation
  * RPSOR
    * PSOR
    * PGS
    * RPGS
  * Bokhoven Iterative Method
* Mixed Linear Complementarity Problems (MLCP)
* Nonlinear Complementarity Problems (NCP)
* Mixed Complementarity Problems (MCP)
  * Nonlinear Reformulation
  * PATH Solver (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl))

## Usage

More details are WIP. Examples can be found in `test` directory.
