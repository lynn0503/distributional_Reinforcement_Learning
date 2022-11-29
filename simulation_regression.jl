using Distributions
using Plots

# test Distributions
# d=Normal()
# x=rand(d,1000)
# histogram(x)

# bimodal gaussian distribution
bimodal=MixtureModel(Normal[Normal(5, 1),Normal(15, 1)], [0.3, 0.7])
x=rand(bimodal,1000)
histogram(x;bins=0:0.5:20)
mean(x)
# simulate traditional RL
α = 0.1
trials = 2000
vals = fill(0,trials)
for t in 2:trials
    δ = reward - val[t-1]
    vals[t] = val[t-1] + α * δ
end

plot(vals)

# simulate distributional RL