include("dcg_scorer.jl")

type NDCGScorer <: AbstractDCGScorer
    k::Int64
    ideal_gains::Dict{Float32, Float32}
    # properties from DCG
    gain::Vector{Float64}
    discount::Vector{Float64}
end

NDCGScorer(k) = NDCGScorer(k, Dict{Float32, Float32}(), Vector{Float64}(), Vector{Float64}())
NDCGScorer() = NDCGScorer(10)
gain(scorer::NDCGScorer) = scorer.gain
discount(scorer::NDCGScorer) = scorer.discount

get_relevance_labels(rl::RankList) = map(dp->Int64(label(dp)), rl)

function score(scorer::NDCGScorer, rank_list::RankList)
    if length(rank_list) == 0
        return 0
    end
    size = scorer.k
    if scorer.k > length(rank_list) || scorer.k <= 0
        size = length(rank_list)
    end
    rel = get_relevance_labels(rank_list)
    ideal = 0.0
    if haskey(scorer.ideal_gains, id(rank_list))
        ideal = scorer.ideal_gains[id(rank_list)]
    else
        ideal = get_ideal_dcg(scorer, rel, size)
        scorer.ideal_gains[id(rank_list)] = ideal
    end

    if ideal <= 0.0
        return 0.0
    end
    get_dcg(scorer, rel, size) / ideal
end

name(scorer::NDCGScorer) = @sprintf("NDCG@%d", scorer.k)
k(scorer::NDCGScorer) = scorer.k

function get_ideal_dcg(scorer::NDCGScorer, rel::Vector{Int64}, top_k::Int64)
    idx = sortperm(rel, rev=true)
    dcg = 0.0
    for i = 1 : top_k
        dcg += gain(scorer, rel[idx[i]]) * discount(scorer, i)
    end
    dcg
end
