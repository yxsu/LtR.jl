
"""
compute geometry mean by Cholesky decomposition
"""
function geometry_mean_chol(A::Matrix, B::Matrix, t::Real, noise::Float64)
  inv_A = inv(A)

  cond_A = cond(inv_A)
  cond_B = cond(B)
  if cond_A > cond_B
    # swap A and B
    tmp = inv_A
    inv_A = B
    B = tmp
  end
  # avoid PosDefException
  for i = 1 : size(inv_A, 1)
      if inv_A[i, i] < noise
          inv_A[i, i] = noise
      end
  end
  for i = 1 : size(B, 1)
      if B[i, i] < noise
          B[i, i] = noise
      end
  end
  chol_A = chol(Hermitian(inv_A))
  chol_B = chol(Hermitian(B))
  Z = chol_B / chol_A
  v, U = eig(Z'*Z)
  T = diagm(complex(v).^(t/2)) * U' * chol_A
  # compute mean
  return real(T'*T)
end
