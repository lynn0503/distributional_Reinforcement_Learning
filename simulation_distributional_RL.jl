using Distributions
using Plots
using StatsBase

# test Distributions
# d=Normal()
# x=rand(d,1000)
# histogram(x)

# bimodal gaussian distribution
bimodal=MixtureModel(Normal[Normal(5, 1),Normal(15, 1)], [0.3, 0.7])
x=rand(bimodal,2000)
histogram(x;bins=0:0.5:20,legend=false,normalize=true)
mean(x)
# set parameters
blocks =10
trials = 2000
# simulate traditional RL
# converge to mean 
α = 0.01:0.01:0.1
vals = fill(0.0,trials,blocks)
for b in 1:blocks
    for t in 2:trials
        δ = x[t-1] - vals[t-1,b]
        vals[t,b] = vals[t-1,b] + α[b] * δ
    end
end
plot(vals,legend=false)
hline!([mean(x)], color=:darkred, linestyle=:dash)

# simulate distributional RL
# converge to quantile
dblocks = 100
α₊ = range(start=0.001,stop=0.1,length=dblocks)
α₋ = range(start=0.1,stop=0.001,length=dblocks)

dvals = fill(0.0,trials,dblocks)

for b in 1:dblocks
    for t in 2:trials
        # println(x[t-1] - dvals[t-1,b])
        if x[t-1] - dvals[t-1,b] > 0
            δ = 1
            dvals[t,b] = dvals[t-1,b] + α₊[b] * δ
        else
            δ = -1
            dvals[t,b] = dvals[t-1,b] + α₋[b] * δ
        end
    end
end

# l = @layout [a;b]
plot(dvals,legend=false)
convergence = dvals[trials,:]
histogram(convergence,bins=0:0.5:20,legend=false)
# plot(a,b,layout=l)
# plot cumulative distribution function
cdf_drl = ecdf(convergence)
plot(x -> cdf_drl(x),0,20, legend=false)
quantiles=0.1:0.1:1
values=quantile(convergence,quantiles)
scatter!(values, quantiles, markersize = 10)
# asymmetric scaling factor 
τ = α₊/(α₊+α₋)
plot(τ,legend=false)
println(τ)

# converge to expectile
# dblocks = 100
# α₊ = range(start=0.001,stop=0.1,length=dblocks)
# α₋ = range(start=0.1,stop=0.001,length=dblocks)
evals = fill(0.0,trials,dblocks)

for b in 1:dblocks
    for t in 2:trials
        δ = x[t-1] - evals[t-1,b]
        # println(δ)
        if  δ > 0
            evals[t,b] = evals[t-1,b] + α₊[b] * δ
        else
            evals[t,b] = evals[t-1,b] + α₋[b] * δ
        end
    end
end

plot(evals,legend=false)

econvergence = evals[trials,:]
histogram(econvergence,bins=0:0.5:20,legend=false,normalize=true)