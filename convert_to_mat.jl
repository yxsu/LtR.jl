fold_name = "Fold1"
large_train_file = joinpath("dataset", "MSLR-WEB10K", fold_name, "train.txt")
large_vali_file = joinpath("dataset", "MSLR-WEB10K", fold_name, "vali.txt")
large_test_file = joinpath("dataset", "MSLR-WEB10K", fold_name, "test.txt")

addprocs()
@everywhere include(joinpath(pwd(), "src", "rank_list.jl"))
#samples = LtR.read_rank_lists(large_train_file)
samples_train = read_rank_lists(large_train_file)
samples_vali = read_rank_lists(large_vali_file)
samples_test = read_rank_lists(large_test_file)

function extract(samples::Vector{RankList})
	num_dp = 0
	for sample in samples
		num_dp += length(sample)
	end
	num_features = length(samples[1][1])
	X = zeros(Float32, (num_features, num_dp))
	labels = zeros(Int32, num_dp)
    qids = zeros(Int32, num_dp)
	index_dp = 1
	for sample in samples
		for dp in LtR.data(sample)
			X[:, index_dp] = dp.features
			labels[index_dp] = LtR.label(dp)
            qids[index_dp] = LtR.id(dp)
			index_dp += 1
		end
	end
	(X, labels, qids)
end
train_features, train_labels, train_qids = extract(samples_train)
vali_features, vali_labels, vali_qids = extract(samples_vali)
test_features, test_labels, test_qids = extract(samples_test)
using MAT
file = matopen(string(fold_name, ".mat"), "w")
write(file, "train_feature", train_features)
write(file, "train_label", train_labels)
write(file, "train_qid", train_qids)
write(file, "vali_feature", vali_features)
write(file, "vali_label", vali_labels)
write(file, "vali_qid", vali_qids)
write(file, "test_feature", test_features)
write(file, "test_label", test_labels)
write(file, "test_qid", test_qids)
close(file)
