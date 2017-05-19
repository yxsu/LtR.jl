
global _max_num_features = 0

_default_batch_size = 1000 # the chunk size when loading dataset

type DataPoint
	label::Float32 # ground thuth, the real label of the data point (e.g. its degree of relevance according to the relevance judgement)
	id::Int64 # id of this data point (e.g. query-id)
	features::Vector{Float32}
end

"""
the valid line should be 'label qid: id feature1:value1 feature2:value2 ...'
"""
function parse_datapoint(text::String)
	global _max_num_features
	text = strip(text)
	pos = searchindex(text, '#')
	if pos > 0
		text = text[1:pos]
	end
	tmp = Base.split(text, ' ')
	label = parse(tmp[1])
	id = parse(tmp[2][5:end]) # extrat from "qid:id"
	if _max_num_features < length(tmp) - 2
		_max_num_features = length(tmp) - 2
	end
	features = Vector{Float32}(_max_num_features)
    for i = 3 : length(tmp)
        str_key, str_value = Base.split(tmp[i], ':')
        #key = parse(str_key)
        value = parse(str_value)
		features[i - 2] = value
    end
	DataPoint(label, id, features)
end

function read_datapoints(filename::String)
	tmp = readlines(filename)
	pmap(parse_datapoint, tmp, batch_size=_default_batch_size)
end

id(p::DataPoint) = p.id
label(p::DataPoint) = p.label
feature_value(p::DataPoint, id::Int) = p.features[id]
Base.length(p::DataPoint) = length(p.features)
