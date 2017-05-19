type DataStore
  y::Vector{String} # The list of labels
  X::Matrix{Float64} # raw data samples

  function DataStore(y::Vector{String}, X::Matrix{Float64})
    new(y, X)
  end
end

type AlgGMML
  lambda::Float64
  t::Float64
  A::Nullable{Matrix{Float64}}
end

AlgGMML(lambda::Float64, t::Float64) = AlgGMML(lambda, t, Nullable())
AlgGMML() = AlgGMML(1.0, 0.5)

# renew the metric in `MetricLearningAlg` to Euclidean space
function renew_metric(alg::AlgGMML, d::Int64)
  alg.A = Nullable(eye(d, d))
end

include("geometry_mean.jl")

function run!(alg::AlgGMML, data::DataStore, similar_pairs::Vector{Pair{Int64, Int64}}, dissimilar_pairs::Vector{Pair{Int64, Int64}})
  X = data.X
  d = size(X, 1)
  if isnull(alg.A)
      alg.A = Nullable(eye(d,d))
  end
  A = get(alg.A)
  S = zeros(d, d)
  D = zeros(d, d)
  # generate constraint matrices
  for (x1_index, x2_index) in similar_pairs
    v = X[:, x1_index] - X[:, x2_index]
    S += v*v'
  end
  for (x1_index, x2_index) in dissimilar_pairs
    v = X[:, x1_index] - X[:, x2_index]
    D += v*v'
  end
  # run Cholesky-Schur method
  noise = 1e-8
  has_exception = true
  while has_exception
      try
          alg.A = Nullable(geometry_mean_chol(S + alg.lambda * A, D + alg.lambda * A, alg.t, noise))
          has_exception = false
      catch e
          if isa(e, Base.LinAlg.PosDefException)
              noise *= 10
          end
      end
  end
end
