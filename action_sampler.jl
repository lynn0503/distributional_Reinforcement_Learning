using Statistics
using Distributions
# argmax

# ϵ-greedy
function ϵ_greedy(dist,n,ϵ)
    avgs=mean(dist,dims=1)
    # println(avgs)
    tmp=rand()
    if tmp>ϵ
        car_idx=argmax(avgs)
        idx=getindex(car_idx,2)
        # println(idx)
    else
        idx=rand(1:n)
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
function thompson(a,b)
    # a is how many times to get reward 1 for each arm
    # b is how many times to get reward 0 for each arm
    distribution_for_each_arm=Beta(a,b)
    sample_for_each_arm=rand(distribution_for_each_arm)
    car_idx=argmax(sample_for_each_arm)
    idx=getindex(car_idx,2)
    return idx
end

# UCB
function ucb(dist,choices_cnt,t,β)
    avgs=mean(dist,dims=1)
    # uncertainty=sqrt(-log(p)/2count(action))
    # here let p=1/t
    uncertainty=.√(log(t)/2*(choices_cnt.+1))
    ucbs=transpose(avgs) + β * uncertainty
    car_idx=argmax(ucbs)
    idx=getindex(car_idx,2)
    return idx
end