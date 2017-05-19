using MAT
include("data_point.jl")

type RankList
	data::Vector{DataPoint}

	function RankList()
		new(Vector{DataPoint}())
	end
end

function RankList(other::RankList, idx::Vector{Int64})
    rank_list = RankList()
    resize!(rank_list.data, length(other))
    for i = 1 : length(idx)
        rank_list[i] = other[idx[i]]
    end
    rank_list
end

function correct_rank_list(old::RankList)
	idx = sortperm(data(old), by=dp->label(dp), rev=true)
	RankList(old, idx)
end

function get_ranking(old::RankList, feature_index::Int32)
	new_rank_list = deepcopy(old)
	sort!(new_rank_list.data, by=x->x.features[feature_index], rev=true)
	new_rank_list
end

data(rl::RankList) = rl.data
id(rl::RankList) = id(rl.data[1])

function Base.push!(rank_list::RankList, p::DataPoint)
	push!(rank_list.data, p)
end

function Base.empty!(rank_list::RankList)
	empty!(rank_list.data)
end

Base.length(rl::RankList) = length(rl.data)

Base.getindex(rank_list::RankList, id::Int64) = Base.getindex(rank_list.data, id)
Base.setindex!(rank_list::RankList, dp::DataPoint, id::Int64) = Base.setindex!(rank_list.data, dp, id)
Base.start(rank_list::RankList) = Base.start(rank_list.data)
Base.done(rank_list::RankList, id::Int64) = Base.done(rank_list.data, id)
Base.next(rank_list::RankList, status::Int64) = Base.next(rank_list.data, status)
feature_size(rl::RankList) = length(rl.data[1])

function read_rank_lists(filename::String)
	dps = convert(Vector{DataPoint}, read_datapoints(filename))
	rank_dict = read_ranklist_dict(dps)
	collect(values(rank_dict))
end

function read_ranklist_dict(dps::Vector{DataPoint})
	rank_dict = Dict{Int32, RankList}()
	for dp in dps
	    if !haskey(rank_dict, id(dp))
	        rank_dict[id(dp)] = RankList()
	    end
	    push!(rank_dict[id(dp)], dp)
	end
	rank_dict
end

# read mat data to samples
function read_sample_dict_from_mat_file(filename::String)
    mat_obj = matread(filename)
    samples = Dict([:train=>Dict(), :vali=>Dict(), :test=>Dict()])
    types = [:train, :vali, :test]
    for m in types
        features = mat_obj[string(m, "_feature")]
        qids = mat_obj[string(m, "_qid")]
        labels = mat_obj[string(m, "_label")]
        dps = Vector{DataPoint}(length(labels))
        for i = 1 : length(labels)
            dps[i] = DataPoint(labels[i], qids[i], features[:, i])
        end
        samples[m] = read_ranklist_dict(dps)
    end
    samples
end

function normalize_samples!(samples::Vector{RankList}, p::Int64)
    for f = 1 : feature_size(samples[1])
        tmp_array = Vector{Float32}()
        for sample in samples
            append!(tmp_array, map(dp->feature_value(dp, f), data(sample)))
        end
        normalize!(tmp_array, p)
        # copy element back
        base = 0
        for sample in samples
            dps = data(sample)
            for j = 1 : length(dps)
                dps[j].features[f] = tmp_array[base + j]
            end
            base += length(dps)
        end
    end
end
