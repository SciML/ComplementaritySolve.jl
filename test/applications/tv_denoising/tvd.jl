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

cd(@__DIR__)

using Pkg
Pkg.activate(".")

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
    StableRNGs,
    Lux
using ComplementaritySolve

rng = StableRNG(0)

N = 100
t = 1:100
temp= ones(Int(ceil((N)/4)))
exact = vcat(temp,-1*temp,temp,-1*temp)
exact = exact[1:N] + 0.5*sin.((2*pi/N)*t)
w = 0.1*rand(rng,N)

seqLen = 100
nF,nH,nIn = seqLen,2*seqLen-1,2*seqLen-2

L_m = tril(ones(Float64,(nH,nH)));
#=Q = 1e-8*Matrix(1.0I,nH,nH)
Q[1:nF,1:nF] = Matrix(1.0I,nF,nF)
L = cholesky(Q).L=#

I_b = -1.0*Matrix(1.0I,nF-1,nF-1);
eps = 1e-4

function θ_init()
    D = 0.3*rand(nF-1,nF)
    Q = 1e-8*Matrix(1.0I,nH,nH)
    Q[1:nF,1:nF] = Matrix(1.0I,nF,nF)
    L = cholesky(Q).L
    h = zeros(Float64,nIn)
    c = 13.0*ones(Float64,nF-1)
    return ComponentArray(;D = D,Q = Q,c = c,h=h,L = L)  
end 

function forward(x,p)
    D,Q,c,h,L = p.D,p.Q,p.c,p.h,p.L
    L = L_m*L
    Q = L * L' + eps*Matrix(1.0I,nH,nH)
    G = [D I_b;
        -D I_b]
    M = [Q  G'; 
       -G  zeros(Float64,nIn,nIn)]
    c = vcat(-x[1:nF],c)
    q = vcat(c,h)
    prob = LCP(M,q,zeros(Float64,size(q,1)))
    solver = NonlinearReformulation()
    sol = solve(prob,solver;sensealg=LinearComplementarityAdjoint())
    return sol.u

end

function create_optimizer(θ)
    opt = Optimisers.ADAM(0.01)
    return Optimisers.setup(opt,θ)
end

function compute_loss(x,θ)
    return sum(abs2,(exact - x))/100.0 + 0.1*norm(θ.D,1)
    #return sol.z[1:100]
end


#do the gradient decent
n_iterations = 1000
x = exact + w
θ_in = θ_init()
opt = create_optimizer(θ_in)

for epoch in 1:n_iterations
    x_temp = forward(x,θ_in)
    loss,back = pullback(p -> compute_loss(x_temp[1:100],p),θ_in)
    gs = back(1)[1]
    opt, θ_in = Optimisers.update(opt, θ_in, gs)
    x = x_temp[1:100]
    println("Epoch [$epoch]: Loss $loss")
end

