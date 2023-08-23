#=
Total variation assumes that given noisy data y is given as follows

y = x + w where x is a piecwise constant signal and w is a gaussian noise. It estimates x by solving optimization problem

argmin `0.5 ||y-x|| + λ||Dx|| `

where D is the different operator where each row of D,D_i = e_i - e_i+1

we are going to try and learn the operator 
=#

"""
Given a quadractic program given in the form

minimize 0.5 x'Qx + c'x subject to x >=0 and Gx<=h

we rewrite it as to LCP(M,q)

M = [Q G'
    -G 0]

q =[c
    h
    ]

LCP solves for z =(x
                   `λ`)
"""

using Zygote,
    LinearAlgebra,
    SimpleNonlinearSolve,
    TruncatedStacktraces,
    OrdinaryDiffEq,
    Plots,
    Optimization,
    OptimizationOptimisers,
    SciMLSensitivity,
    Test,
    ComponentArrays,
    StableRNGs

using ComplementaritySolve
using NonlinearSolve
using PyCall
using PyPlot
#using FiniteDiff,Symbolics,SparseDiffTools,StableRNGs,Test
#using NonlinearSolve
#using BenchmarkTools,StableRNGs
using PATHSolver
using FiniteDiff

PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")

rng = StableRNG(0)

torch = pyimport("torch")



N = 100
#t = 1:N
#temp= ones(Int(ceil((N)/4)))
exact = Float64.(vec(torch.load("/home/yonatanwesen/optnet/denoising/data/synthetic/labels.pt").numpy()))#vcat(temp,-1*temp,temp,-1*temp)
#exact = exact[1:N] + 0.5*sin.((2*pi/N)*t)
#w = 1.0*rand(rng,N)

seqLen = N
nF,nH,nIn = seqLen,2*seqLen-1,2*seqLen-2

L_m = tril(ones(Float64,(nH,nH)));
#=Q = 1e-8*Matrix(1.0I,nH,nH)
Q[1:nF,1:nF] = Matrix(1.0I,nF,nF)
L = cholesky(Q).L=#

I_b = -1.0*Matrix(1.0I,nF-1,nF-1);
eps = 1e-4

function θ_init()
    #D = 0.3*randn(rng,nF-1,nF)
    D = zeros(Float64,nF-1,nF)
    D[1:nF-1,1:nF-1] = Matrix(1.0I,nF-1,nF-1)
    D[1:nF-1,2:nF] -= Matrix(1.0I,nF-1,nF-1)
    #D +=randn(nF-1,nF)
    Q = 1e-8*Matrix(1.0I,nH,nH)
    Q[1:nF,1:nF] = Matrix(1.0I,nF,nF)
    L = cholesky(Q).L
    z = zeros(Float64,nH)
    s = ones(Float64,nIn)
    return ComponentArray(;Q = Q,s=s,z=z,L = L)  
end 

function forward(p)
    Q,z,s,L = p.Q,p.z,p.s,p.L
    L = L_m*L
    Q = L * L' + eps*Matrix(1.0I,nH,nH)
    #=
     G = [D I_b
         -D I_b]
    =#
    G = vcat(hcat(D,I_b),hcat(-1.0*D,I_b))
    #=
    M = [Q'  G'
       -G  zeros(Float64,nIn,nIn)]
    =#
    M =vcat(hcat(Q,Transpose(G)),hcat(-1.0*G,zeros(Float64,nIn,nIn))) 
    c = zeros(Float64,2*nF-1)#vcat(-x,13.0*ones(Float64,nF-1))
    h= G*z + s
    q = vcat(c,h)
    prob =LCP(M,q)
    solver = NonlinearReformulation(:smooth, NewtonRaphson())#InteriorPointMethod()
    sol = solve(prob,solver;sensealg=LinearComplementarityAdjoint())
    return sol.u

end

function _forward(x,p)
    D,Q,h,L = p.D,p.Q,p.h,p.L
    L = L_m*L
    Q = L * L' + eps*Matrix(1.0I,nH,nH)
    #=
     G = [D I_b;
         -D I_b]
    =#
    G = vcat(hcat(D,I_b),hcat(-1.0*D,I_b))
    #=
    M = [Q'  G'; 
       -G  zeros(Float64,nIn,nIn)]
    =#
    M =vcat(hcat(Q,-1.0*Transpose(G)),hcat(G,zeros(Float64,nIn,nIn))) 
    c = vcat(-x,13.0*ones(Float64,nF-1))
    q = vcat(c,h)
    prob = MCP(LCP(M,q))#LCP(M,q)#MCP(LCP(M,q))
    #println(prob.f)
    solver = PATHSolverAlgorithm()#NonlinearReformulation()
    sol = solve(prob,solver;sensealg=MixedComplementarityAdjoint(),verbose =false)
    return sol.u
end


function create_optimizer(θ,learning_rate)
    opt = Optimisers.ADAM(learning_rate)
    return Optimisers.setup(opt,θ)
end

function compute_loss(x,θ)
    return sum(abs2,(exact - x))/N 
    #return sol.z[1:100]
end

function sort_D(after_D)
    D_a = after_D.^6
    n = size(D_a)[2]
    D_a = D_a./sum(D_a,dims=2)
    I = sortperm(D_a*vec(0:n-1))
    return I
end

function plot_D(initD,latestD)
    function p(D,fname)
        PyPlot.clf()
        lim = max(abs(minimum(D)),abs(maximum(D)))
        cli = (-lim,lim)
        PyPlot.imshow(D,cmap="bwr",interpolation ="nearest",clim=cli)
        PyPlot.colorbar()
        PyPlot.savefig(fname)
    end

    p(initD,"initD-offset.png")
    p(latestD,"latestD-offset.png")
     
end


#do the gradient decent
#begin
learning_rate = 0.01
n_iterations = 5000
x = Float64.(vec(torch.load("/home/yonatanwesen/optnet/denoising/data/synthetic/features.pt").numpy()))#exact + w
θ_in = θ_init()
#D_before = θ_in.D
opt = create_optimizer(θ_in,learning_rate)
#D_norms = [norm(θ_in.D,1)]
#min_loss = Inf64
#x_min = copy(x) 
#θ_min = copy(θ_in)
for epoch in 1:n_iterations
    x_temp = forward(θ_in)
    loss,back = pullback(p -> compute_loss(x_temp[1:N],p),θ_in)
    gs = back(1)[1]
    opt, θ_in = Optimisers.update(opt, θ_in, gs)
    #=push!(D_norms,norm(θ_in.D,1))
    if loss < min_loss
        min_loss = loss
        x_min = x_temp[1:N]
        θ_min = copy(θ_in)
    end=#

    x = x_temp[1:N]
    #=if epoch in (72,73)
        @show x
    end=#
    if loss < 0.1
        break
    end
    println("Epoch [$epoch]: Loss $loss")
end #catch e end
#end
opt = create_optimizer(θ_in,0.002)
n_iterations = 200
for epoch in 1:n_iterations
    x_temp = forward(x,θ_in)
    loss,back = pullback(p -> compute_loss(x_temp[1:N],p),θ_in)
    gs = back(1)[1]
    opt, θ_in = Optimisers.update(opt, θ_in, gs)
    #=push!(D_norms,norm(θ_in.D,1))
    if loss < min_loss
        min_loss = loss
        x_min = x_temp[1:N]
        θ_min = copy(θ_in)
    end=#

    x = x_temp[1:N]
    #=if epoch in (72,73)
        @show x
    end=#
    if loss < 0.1
        break
    end
    println("Epoch [$epoch]: Loss $loss")
end


#testing some stuff
#θ_in = θ_init()
loss,back = pullback(p -> compute_loss(x,p),θ_in)
gs = back(1)[1]
g = FiniteDiff.finite_difference_gradient(p -> compute_loss(x,p),θ_in)
g.D ≈ gs.D

sol = forward(x,θ_in)
sol_p = _forward(x,θ_in)

