addprocs()
@everywhere include(joinpath("src", "rank_list.jl"))
@everywhere include(joinpath("src", "lgmml.jl"))
filename = joinpath("dataset", "train_subset.txt")
print("Read rank lists from $(filename)...")
samples = read_rank_lists(filename)
print("finished\n")
print("normalizing data...")
normalize_samples!(samples, 2)
print("finished\n")

m = 2
sampling_rate = 0.8
scorer = NDCGScorer()

#learn_metric_and_anchor(samples[1], NDCGScorer(), sampling_rate)

lgmml = LGMML(samples, m, sampling_rate)

train(lgmml, samples, 5000)
