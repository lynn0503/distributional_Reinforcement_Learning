using Statistics
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
function thompson(dist)

end

# UCB
function ucb(dist,β,ϵ)
    avgs=mean(dist,dims=1)
    stds=std(dist,dims=1)
    ucbs=avgs + β * stds
    tmp=rand()
    if tmp>ϵ
        car_idx=argmax(ucbs)
        idx=getindex(car_idx,2)
    else
        idx=rand(1:n)
    end
    
    return idx
end