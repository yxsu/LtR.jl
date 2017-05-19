abstract AbstractDCGScorer

type DCGScorer <: AbstractDCGScorer
    #parent
    k::Int32
    #local
    discount::Vector{Float32}
    gain::Vector{Float32}

    function DCGScorer()
        discount = Vector{Float32}(5000)
        for i = 1 : length(discount)
            discount[i] = 1.0 / log2(i + 1)
        end
        gain = Vector{Float32}(6)
        for i = 1 : 6
            gain[i] = (1<<(i-1)) - 1
        end
        new(10, discount, gain)
    end
end
gain(scorer::DCGScorer) = scorer.gain
discount(scorer::DCGScorer) = scorer.discount

function get_dcg(scorer::AbstractDCGScorer, rel::Vector{Int64}, top_k::Int64)
    dcg = 0.0
    for i = 1 : top_k
        dcg += gain(scorer, rel[i]) * discount(scorer, i)
    end
    dcg
end

function score(scorer::DCGScorer, rank_list::RankList)
    if length(rank_list) == 0
        return 0
    end
    size = scorer.k
    if length(rank_list) < k || k<=0
        size = length(rank_list)
    end
    rel = get_relevance_labels(rank_list)
    get_dcg(scorer, rel, size)
end

function discount(scorer::AbstractDCGScorer, index::Int64)
    if index <= length(discount(scorer))
        return discount(scorer)[index]
    end
    # expand
    cache_size = length(discount(scorer)) + 1000
    while cache_size < index
        cache_size += 1000
    end
    old_size = length(discount(scorer))
    resize!(discount(scorer), cache_size)
    for i = old_size + 1 : cache_size
        discount(scorer)[i] = 1.0 / log2(i + 1)
    end
    discount(scorer)[index]
end

function gain(scorer::AbstractDCGScorer, rel::Int64)
    rel = rel + 1
    if rel <= length(gain(scorer))
        return gain(scorer)[rel]
    end
    # expand
    cache_size = length(gain(scorer)) + 10
    while cache_size < rel
        cache_size += 10
    end
    old_size = length(gain(scorer))
    resize!(gain(scorer), cache_size)
    for i = old_size + 1 : cache_size
        gain(scorer)[i] = 1<<(i-1) - 1
    end
    gain(scorer)[rel]
end
