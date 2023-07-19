using Convex,SCS
using Plots

# length of the signal
N = 100
t = 0:100
temp= ones(Int(ceil((N+1)/4)))
exact = vcat(temp,-1*temp,temp,-1*temp)
exact = exact[1:N+1] + 0.5*sin.((2*pi/N)*t)

plot(t,exact)

w = 0.1*rand(N+1)
y = exact + w

plot!(t,y)

z_1 = Variable(N+1)
z_2 = Variable(N+1)
z_3 = Variable(N+1)

lam_1 = 10.0
lam_2 = 8.0
lam_3 = 5.0


obj_1 = 0.0

obj_1 += 0.5 *sumsquares(y-z_1)
obj_1 += (lam_1*sum(abs.(y[2:N+1] - y[1:N])))

obj_2 = 0.0

obj_2 += 0.5 *sumsquares(y-z_2)
obj_2 += (lam*sum(abs.(y[2:N+1] - y[1:N])))

obj_3 = 0.0

obj_3 += 0.5 *sumsquares(y-z_3)
obj_3 += (lam*sum(abs.(y[2:N+1] - y[1:N])))



prob1 = minimize(obj_1)
solve!(prob1,SCS.Optimizer)

prob2 = minimize(obj_2)
solve!(prob2,SCS.Optimizer)

prob3 = minimize(obj_3)
solve!(prob3,SCS.Optimizer)

plot!(t,z_1.value)
plot!(t,z_2.value)
plot!(t,z_2.value)