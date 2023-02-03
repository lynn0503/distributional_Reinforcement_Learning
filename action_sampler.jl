using Statistics
using Distributions
# argmax

# ϵ-greedy
function ϵ_greedy(vals,ϵ,K)
    tmp=rand()
    if tmp>ϵ
        idx=argmax(vals)
    else
        idx=rand(1:K)
    end
    return idx
end

function decay_ϵ_greedy(vals,ϵ,t,K)
    ϵ=ϵ/t
    tmp=rand()
    if tmp>ϵ
        idx=argmax(vals)
    else
        idx=rand(1:K)
    end
    return idx
end

function naive_bonus(dist,c,ϵ,K)
    tmp=rand()
    if tmp>ϵ
        avgs=mean(dist,dims=1)
        stds=std(dist,dims=1)
        vals=avgs+c.*stds
        car_idx=argmax(vals)
        idx=getindex(car_idx,2)
    else 
        idx=rand(1:K)
    end
    return idx
end

function vanish_bonus(dist,c,t,ϵ,K)
    tmp=rand()
    if tmp>ϵ
        avgs=mean(dist,dims=1)
        stds=std(dist,dims=1)
        ct=c*sqrt(log(t)/t)
        vals=avgs+ct.*stds
        car_idx=argmax(vals)
        idx=getindex(car_idx,2)
    else 
        idx=rand(1:K)
    end
    return idx
end

function truncated_variance(dist,c,t,ϵ,K)
    N=size(dist,1)
    tmp=rand()
    if tmp>ϵ
        sort!(dist,dims=1)
        avgs=mean(dist,dims=1)
        mid=floor(Int,N/2)
        stds_left=std(dist[mid:N,:])
        ct=c*sqrt(log(t)/t)
        vals=avgs.+ct*stds_left
        car_idx=argmax(vals)
        idx=getindex(car_idx,2)
    else
        idx=rand(1:K)
    end
    return idx
end
# softmax
function sftmax(dist,β)
    avgs=mean(dist,dims=1)
    soft_prob=exp.(β.*avgs)./sum(exp.(β.*avgs))
    idx=sum(cumsum(soft_prob,dims=2).<rand())+1
    return idx
end
# test softmax
# idx_cnt=zeros(3)
# for i in 1:1000
#     idx=sum(cumsum([0.3,0.5,0.2]).<rand())+1
#     idx_cnt[idx]+=1
# end
# bar(idx_cnt)

# Thompson sampling
function thompson(values_trial_t)
    # for distributional RL no need to infer reward distribution
    # because neuronal population encode this distribution
    # generate one sample for one arm from this distribution
    samples_idx=rand(1:100,2)
    # println(samples_idx)
    samples=values_trial_t[samples_idx,[1,2]]
    cart_idx=argmax(samples)
    idx=getindex(cart_idx,2)
    return idx
end

# UCB
function ucb(dist,choices_cnt,t,β)
    avgs=mean(dist,dims=1)
    # uncertainty=sqrt(-log(p)/2count(action))
    # here let p=1/t
    uncertainty=.√(log(t)/2*(choices_cnt.+1))
    ucbs=transpose(avgs) + β * uncertainty
    cart_idx=argmax(ucbs)
    idx=getindex(cart_idx,2)
    return idx
end

