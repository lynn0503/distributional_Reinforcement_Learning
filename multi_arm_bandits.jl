using Distributions
using Plots
using Random
include("action_sampler.jl")

# Random.seed!(2020)
n=2
trials=1000
arm1=Normal(10,2)
arm1_reward=rand(arm1,trials)
arm2=Normal(10,1)
arm2_reward=rand(arm2,trials)
# arm3=Normal(20,2)
# arm3_reward=rand(arm3,trials)
# rewards=hcat(arm1_reward,arm2_reward,arm3_reward)
rewards=hcat(arm1_reward,arm2_reward)
size(rewards)
reward_agent=zeros(trials)
# distributional Rascolar Wagner
# assign a pair of α₊ and α₋ for each neuron
population = 100
α₊ = range(start=0.01,stop=0.1,length=population)
α₋ = range(start=0.1,stop=0.01,length=population)
# maintain a distributional value matrix for each arm
values=zeros(n,population,trials)
values[1,:,1]=ones(population)*10*rand()
values[2,:,1]=ones(population)*10*rand()
# values[3,:,1]=ones(population)*rand()*10
# initialize choices for all trials
choices=zeros(Int,trials)
choices[1]=rand(1:n)
# simulation
choices_cnt=zeros(Int,2)
choices_cnt[choices[1]]+=1
for t in 1:trials-1
    # reward to agent
    choice=choices[t]
    # println("-------------------------")
    reward=rewards[t,choice]
    reward_agent[t]=reward
    
    # update value for the current choice
    for p in 1:population
        # track number of past selection
        trial_arm_i=choices_cnt[choice]
        if trial_arm_i==0
            trial_arm_i=1
        end
        value=values[choice,p,trial_arm_i]
        δ=reward-value
        if δ>0
            δ=1
            new_value = value + α₊[p] * δ
        else
            δ=-1
            new_value = value + α₋[p] * δ
        end
        values[choice,p,trial_arm_i+1]=new_value
    end
    # select new choice with action_sampler
    trial_arm_1=choices_cnt[1]
    trial_arm_2=choices_cnt[2]
    # trial_arm_3=choices_cnt[3]
    if trial_arm_1==0
        trial_arm_1=1
    end
    if trial_arm_2==0
        trial_arm_2=1
    end
    # if trial_arm_3==0
    #     trial_arm_3=1
    # end
    # values_trial_t=hcat(values[1,:,trial_arm_1],values[2,:,trial_arm_2],values[3,:,trial_arm_3])
    values_trial_t=hcat(values[1,:,trial_arm_1],values[2,:,trial_arm_2])
    # println(size(values_trial_t))
    # size 100*3
    # new_choice=ϵ_greedy(values_trial_t,n,0.1)
    # new_choice=sftmax(values_trial_t,0.1)
    new_choice=ucb(values_trial_t,choices_cnt,t,0.1)
    # println(new_choice)
    choices[t+1]=new_choice
    choices_cnt[new_choice]+=1
    if t==trials-1
        global convergence=values_trial_t
    end
end

histogram(rewards,normalize=true,alpha=0.3)
histogram!(convergence,normalize=true,alpha=0.3)

histogram(rewards[:,1],normalize=true,alpha=0.3)
histogram!(convergence[:,1],normalize=true,alpha=0.3)

histogram(rewards[:,2],normalize=true,alpha=0.3)
histogram!(convergence[:,2],normalize=true,alpha=0.3)
# plot(cumsum(reward_agent))
# plot(reward_agent)
histogram(choices)
# println("first choice is :",choices[1])
# println("freqency of first choice is:", sum(choices.==choices[1])/trials)