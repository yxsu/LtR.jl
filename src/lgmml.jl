using Iterators
include("ndcg_scorer.jl")
include("gmml.jl")

type LGMML
  scorer::NDCGScorer
  anchors::Vector{Vector{Float64}} # d \times n (number of anchor points)
  metrics::Vector{Matrix{Float64}} # set of matrices
  weights::Dict{Int64, Vector{Float64}} # query-id -> weight default 1 for all local metrics
end

LGMML() = LGMML(0, Matrix{Float64}(), Vector{Matrix{Float64}}(), Dict)

function LGMML(samples::Vector{RankList}, m::Int64, sampling_rate::Float64; scorer::NDCGScorer = NDCGScorer())
  print("computing anchors and local metrics...")
  meta_results = pmap(s->learn_metric_and_anchor(s, scorer, sampling_rate), random_subset(samples, m))
  print("finishe\n")
  anchors = Vector{Vector{Float64}}()
  metrics = Vector{Matrix{Float64}}()
  for meta in meta_results
    push!(anchors, meta[2].features)
    push!(metrics, meta[3])
  end
  LGMML(scorer, anchors, metrics, Dict{Int64, Vector{Float64}}())
end

function train(lgmml::LGMML, training_samples::Vector{RankList}, rounds::Int64)
  num_samples = length(training_samples)
  total_iterations = rounds * num_samples
  mu = 1.0
  for t = 1 : total_iterations
    i = rand(1:num_samples)
    mu = 1 / (1 + exp(t / ( 100 * num_samples)))
    lgmml_gradient(lgmml, mu, training_samples[i])
    if t % 500 == 0
      scores = pmap(rl->score(lgmml, rl), training_samples)
      println("$(name(lgmml.scorer)) : $(mean(scores))")
    end
  end
end

dims(lgmml::LGMML) = length(lgmml.anchors[1])

function score(lgmml::LGMML, rl::RankList)
  idx = sortperm(data(rl), by=dp->evaluate(lgmml, dp))
  sorted_rl = RankList(rl, idx)
  score(lgmml.scorer, sorted_rl)
end

function weights(lgmml::LGMML, qid::Int64)
  if !haskey(lgmml.weights, qid)
    lgmml.weights[qid] = ones(Float64, length(lgmml.anchors))
  end
  lgmml.weights[qid]
end

function update(lgmml::LGMML, new_weights::Vector{Float64}, qid::Int64)
  @assert length(lgmml.anchors) == length(new_weights)
  lgmml.weights[qid] = new_weights
end

function evaluate(lgmml::LGMML, dp::DataPoint)
  @assert length(dp) == dims(lgmml)
  n = length(lgmml.anchors)
  weight = weights(lgmml, id(dp))
  p = dp.features
  local_dists = Vector{Float64}(n)
  # compute local distances
  for i = 1 : n
    v = lgmml.anchors[i] - p
    local_dists[i] = sqrt((v' * lgmml.metrics[i] * v)[1])
  end
  # compute the final distance
  weights_dist = map(p-> p[1] * exp(- p[2]), zip(weight, local_dists))
  - sqrt(sum(map(p-> p[1] * p[2], zip(weights_dist, local_dists))))
end

function random_subset(samples::Vector{RankList}, m::Int64)
  random_index = Set{Int64}()
  while length(random_index) < m
    push!(random_index, rand(1:length(samples)))
  end
  result_samples = Vector{RankList}(m)
  for (i, index) in enumerate(random_index)
    result_samples[i] = samples[index]
  end
  result_samples
end


function ndcg_rank_loss(num::Int64)
    sum = 0
    for i = 1 : num
        sum += 1 / (log2(i + 1))
    end
    sum
end

function lgmml_gradient(lgmml::LGMML, mu::Float64, rl::RankList)
    dps = shuffle(data(rl))
    dps_zero_relevant = filter(dp->label(dp) == 0.0, dps)
    max_label = maximum(map(dp->label(dp), dps))
    pos_index = find(dp->label(dp) == max_label, dps)[1]
    pos = dps[pos_index]
    # find violator
    weight = weights(lgmml, id(rl))
    pos_value = evaluate(lgmml, pos)
    vio_index = 1
    for i = 1 : length(dps_zero_relevant)
        if (evaluate(lgmml, dps_zero_relevant[i]) + 0.001) > pos_value
            vio_index = i
            break
        end
    end
    vio = dps_zero_relevant[vio_index]
    coeff = mu * ndcg_rank_loss(Int(round(length(dps_zero_relevant) / vio_index)))
    # update the weights
    for i = 1 : length(weight)
        v1 = pos.features - lgmml.anchors[i]
        v2 = vio.features - lgmml.anchors[i]
        dist1 = sqrt((v1' * lgmml.metrics[i] * v1)[1])
        dist2 = sqrt((v2' * lgmml.metrics[i] * v2)[1])
        weight[i] = weight[i] - coeff * ( exp(- dist2) * dist2 - exp(- dist1) * dist1 )
        if weight[i] < 0
          weight[i] = 0
        end
    end
    update(lgmml, weight, id(rl))
    weight
end


function learn_metric_and_anchor(rl::RankList, scorer::NDCGScorer, sampling_rate::Float64)
  # obtain ideal rank list
  idx = sortperm(data(rl), by=dp->label(dp), rev=true)
  ideal_sorted_rl = RankList(rl, idx)
  n = length(ideal_sorted_rl)
  d = feature_size(ideal_sorted_rl)
  y = Vector{String}(n)
  X = Matrix{Float64}(d, n)
  for (index, dp) in enumerate(data(rl))
      y[index] = string(Int(label(dp)))
      X[:, index] = dp.features
  end
  # create similar and dissimilar pairs
  labels = [x for x in distinct(y)]
  median_pos = 0
  for i  = 1 : div(length(labels), 2)
    median_pos += count(label->label==labels[i], y)
  end
  end_pos = n - count(label->label=="0", y)
  indexes_high = [x for x in 1:median_pos]
  indexes_low = [x for x in end_pos:n]
  #
  similar_pairs = shuffle([Pair(p[1], p[2]) for p in subsets(indexes_high, 2)])
  dissimilar_pairs = shuffle([Pair(p[1], p[2]) for p in product(indexes_high, indexes_low)])
  num_samples = Int64(round(sampling_rate * min(length(similar_pairs), length(dissimilar_pairs))))
  similar_pairs = similar_pairs[1:num_samples]
  dissimilar_pairs = dissimilar_pairs[1:num_samples]
  #
  alg_gmml = AlgGMML(0.1, 1.0)
  raw_data = DataStore(y, X)
  renew_metric(alg_gmml, d)
  run!(alg_gmml, raw_data, similar_pairs, dissimilar_pairs)
  A = get(alg_gmml.A)
  num_highest_points = count(label->label==labels[1], y)
  max_score = 0.0
  max_index = 0
  for base_index = 1 : num_highest_points
    dp = data(rl)[base_index]
    p = dp.features
    idx = sortperm(data(rl), by=dp->((p - dp.features)'*A*(p - dp.features))[1])
    sorted_rl = RankList(rl, idx)
    s = score(scorer, sorted_rl)
    if s > max_score
      max_score = s
      max_index = base_index
    end
  end
  max_score, data(rl)[max_index], A
end
