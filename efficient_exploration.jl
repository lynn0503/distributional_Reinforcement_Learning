using Distributions
using Plots
using StatsBase
using Random
include("action_sampler.jl")

# Random.seed!(1234)
# parameters
K=10
# K arms
N=50
# N quantiles
c=10
α=0.01
ϵ=0.05
trials=1000
runs=1000
# environment setup
# avg_distribution=Normal(0,1)
# avg_K=rand(avg_distribution,K)
avg_K=randn(K)
best_arm=argmax(avg_K)
std_K=ones(K)*5
std_K[best_arm]=1

function step(avg_K,std_K,action)
    global best_arm
    reward_dist=Normal(avg_K[action],std_K[action])
    reward=rand(reward_dist)
    reward_max=rand(Normal(avg_K[best_arm],std_K[best_arm]))
    regret=reward_max-reward
    return reward,regret
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
        δ=reward-dist[n,action]
        if δ>0
            δ=1
            dist[n,action]=dist[n,action]+α₊[n]*δ
        elseif δ<0
            δ=-1
            dist[n,action]=dist[n,action]+α₋[n]*δ
        end
    end
    return dist
end
# simulate
# initialize

α₊ = range(start=0.01,stop=0.1,length=N)
α₋ = range(start=0.1,stop=0.01,length=N)

rewards_traditional=zeros(trials,runs)
rewards_traditional_decay=zeros(trials,runs)
rewards_naive_bonus=zeros(trials,runs)
rewards_vanish_bonus=zeros(trials,runs)
rewards_truncated_variance=zeros(trials,runs)

regret_traditional=zeros(trials,runs)
regret_traditional_decay=zeros(trials,runs)
regret_naive_bonus=zeros(trials,runs)
regret_vanish_bonus=zeros(trials,runs)
regret_truncated_variance=zeros(trials,runs)

# run simulation
for r in 1:runs
    global value=randn(K)
    global value0d=randn(K)
    global dist1=randn(N,K)
    global dist2=randn(N,K)
    global dist3=randn(N,K)

    for t in 1:trials
        action0=ϵ_greedy(value,ϵ,K)
        rewards_traditional[t,r],regret_traditional[t,r]=step(avg_K,std_K,action0)
        value=value_estimate(action0,rewards_traditional[t,r],value,α)

        action0d=decay_ϵ_greedy(value0d,ϵ,t,K)
        rewards_traditional_decay[t,r],regret_traditional_decay[t,r]=step(avg_K,std_K,action0d)
        value0d=value_estimate(action0d,rewards_traditional_decay[t,r],value0d,α)

        action1=naive_bonus(dist1,c,ϵ,K)
        rewards_naive_bonus[t,r],regret_naive_bonus[t,r]=step(avg_K,std_K,action1)
        dist1=distribution_estimate(action1,rewards_naive_bonus[t,r],dist1,α₊,α₋)

        action2=vanish_bonus(dist2,c,t,ϵ,K)
        rewards_vanish_bonus[t,r],regret_vanish_bonus[t,r]=step(avg_K,std_K,action2)
        dist2=distribution_estimate(action2,rewards_vanish_bonus[t,r],dist2,α₊,α₋)

        action3=truncated_variance(dist3,c,t,ϵ,K)
        rewards_truncated_variance[t,r],regret_truncated_variance[t,r]=step(avg_K,std_K,action3)
        dist3=distribution_estimate(action3,rewards_truncated_variance[t,r],dist3,α₊,α₋)
    end
end

# plot result
l=@layout [p1;p2]
width=1200
height=900

trial_mat=repeat(1:trials,1,runs)
avg0=mean(cumsum(rewards_traditional,dims=1)./trial_mat,dims=2)
avg0d=mean(cumsum(rewards_traditional_decay,dims=1)./trial_mat,dims=2)

avg1=mean(cumsum(rewards_naive_bonus,dims=1)./trial_mat,dims=2)
avg2=mean(cumsum(rewards_vanish_bonus,dims=1)./trial_mat,dims=2)
avg3=mean(cumsum(rewards_truncated_variance,dims=1)./trial_mat,dims=2)

p1=plot(avg0,label="ϵ greedy",legend=:topleft,xlabel="trials",ylabel="average reward",size=(width,height))
plot!(avg0d,label="decay ϵ greedy")
plot!(avg1,label="naive bonus")
plot!(avg2,label="vanish bonus")
plot!(avg3,label="truncated variance")

regret0=mean(cumsum(regret_traditional,dims=1),dims=2)
regret0d=mean(cumsum(regret_traditional_decay,dims=1),dims=2)

regret1=mean(cumsum(regret_naive_bonus,dims=1),dims=2)
regret2=mean(cumsum(regret_vanish_bonus,dims=1),dims=2)
regret3=mean(cumsum(regret_truncated_variance,dims=1),dims=2)

p2=plot(regret0,label="ϵ greedy",legend=:topleft,xlabel="trials",ylabel="average regret",size=(width,height))
plot!(regret0d,label="decay ϵ greedy")
plot!(regret1,label="naive bonus")
plot!(regret2,label="vanish bonus")
plot!(regret3,label="truncated variance")

plot(p1,p2,layout=l)
# bar(avg_K,alpha=0.5)
# bar!(value,alpha=0.5)
# bar!(value0d,alpha=0.5)