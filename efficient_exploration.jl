using Distributions
using Plots
using StatsBase
using Random
include("action_sampler.jl")

# Random.seed!(1234)
# parameters
K=10
N=20
c=10
α=0.01
ϵ=0.05
trials=100
runs=1000
# environment setup
avg_distribution=Normal(10,5)
avg_K=rand(avg_distribution,K)
best_arm=argmax(avg_K)
std_K=ones(K)*5
std_K[best_arm]=1

function step(avg_K,std_K,action)
    reward_dist=Normal(avg_K[action],std_K[action])
    reward=rand(reward_dist)
    return reward
end
# traditional
function value_estimate(action,reward,value,α)
    δ=reward-value[action]
    value[action]=value[action]+α*δ
    return value
end
# distributional
function distribution_estimate(action,reward,dist,α₊,α₋)
    N=size(dist,1)
    for n in 1:N
        δ=reward-dist[action]
        if δ>0
            δ=1
            dist[action]=dist[action]+α₊[n]*δ
        elseif δ<0
            δ=-1
            dist[action]=dist[action]+α₋[n]*δ
        end
    end
    return dist
end
# simulate
# initialize
let
α₊ = range(start=0.01,stop=0.1,length=N)
α₋ = range(start=0.1,stop=0.01,length=N)
rewards_naive_bonus=zeros(trials,runs)
rewards_vanish_bonus=zeros(trials,runs)
rewards_truncated_variance=zeros(trials,runs)
rewards_traditional=zeros(trials,runs)

# run simulation
for r in 1:runs
    value=rand(K)
    dist1=rand(N,K)
    dist2=rand(N,K)
    dist3=rand(N,K)

    for t in 1:trials
        action0=ϵ_greedy(value,ϵ)
        rewards_traditional[t,r]=step(avg_K,std_K,action0)
        value=value_estimate(action0,rewards_traditional[t],value,α)

        action1=naive_bonus(dist1,c,ϵ)
        rewards_naive_bonus[t,r]=step(avg_K,std_K,action1)
        dist1=distribution_estimate(action1,rewards_naive_bonus[t],dist1,α₊,α₋)

        action2=vanish_bonus(dist2,c,t,ϵ)
        rewards_vanish_bonus[t,r]=step(avg_K,std_K,action2)
        dist2=distribution_estimate(action2,rewards_vanish_bonus[t],dist2,α₊,α₋)

        action3=truncated_variance(dist3,c,t,ϵ)
        rewards_truncated_variance[t,r]=step(avg_K,std_K,action3)
        dist3=distribution_estimate(action3,rewards_truncated_variance[t],dist3,α₊,α₋)
    end
end
trial_mat=repeat(1:trials,1,runs)
avg0=mean(cumsum(rewards_traditional,dims=1)./trial_mat,dims=2)
avg1=mean(cumsum(rewards_naive_bonus,dims=1)./trial_mat,dims=2)
avg2=mean(cumsum(rewards_vanish_bonus,dims=1)./trial_mat,dims=2)
avg3=mean(cumsum(rewards_truncated_variance,dims=1)./trial_mat,dims=2)

plot(avg0,label="traditional",legend=:bottomright,xlabel="trials",ylabel="average reward")
plot!(avg1,label="naive bonus")
plot!(avg2,label="vanish bonus")
plot!(avg3,label="truncated variance")
end