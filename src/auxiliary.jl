using LinearAlgebra

function _addone!(a::AbstractMatrix)
    view(a, diagind(a)) .= view(a, diagind(a)) .+ 1
    return a
end
function _subtractone!(a::AbstractMatrix)
    view(a, diagind(a)) .= view(a, diagind(a)) .- 1
    return a
end

# _stiefelexp(W, A, Z, α)
# given an isometry W, and a Stiefel tangent vector Δ = W*A + Z, compute the building blocks
# W′, Q′, R′ of the geodesic with respect to the canonical metric in the direction of α*Δ.
# Here, W′is the new isometry, and the local tangent vector is given by
# Δ′ = W′ * A + Z′ with Z′ = Q′*R′
# Here, Q′ is a set of orthogonal columns to the colums in W′.
function _stiefelexp(W::StridedMatrix, A::StridedMatrix, Z::StridedMatrix, α)
    n, p = size(W)
    r = min(2 * p, n)
    QQ, _ = LinearAlgebra.qr!([W Z])
    Q = similar(W, n, r - p)
    @inbounds for j in Base.OneTo(r - p)
        for i in Base.OneTo(n)
            Q[i, j] = (i == p + j)
        end
    end
    Q = lmul!(QQ, Q)
    R = Q' * Z
    A2 = similar(A, min(2 * p, n), min(2 * p, n))
    A2[1:p, 1:p] .= α .* A
    A2[(p + 1):end, 1:p] .= α .* R
    A2[1:p, (p + 1):end] .= (-α) .* (R')
    A2[(p + 1):end, (p + 1):end] .= 0
    U = [W Q] * exp(A2)
    U = project_isometric!(U)
    W′ = U[:, 1:p]
    Q′ = U[:, (p + 1):end]
    R′ = R
    return W′, Q, Q′, R′
end

function _stiefellog(Wold::StridedMatrix, Wnew::StridedMatrix;
                     tol=10 * scalareps(Wold), maxiter=100)
    n, p = size(Wold)
    r = min(2 * p, n)
    P = Wold' * Wnew
    dW = Wnew - Wold * P
    QQ, _ = LinearAlgebra.qr!([Wold dW])
    Q = similar(Wold, n, r - p)
    @inbounds for j in Base.OneTo(r - p)
        for i in Base.OneTo(n)
            Q[i, j] = (i == p + j)
        end
    end
    Q = lmul!(QQ, Q)
    R = Q' * dW
    F = LinearAlgebra.qr!([P; R])
    U = lmul!(F.Q, MatrixAlgebraKit.one!(similar(P, r, r)))
    U[1:p, 1:p] .= P
    U[(p + 1):r, 1:p] .= R
    X = view(U, 1:p, (p + 1):r)
    Y = view(U, (p + 1):r, (p + 1):r)
    if p < n
        USVᴴ = svd_compact!(Y)
        mul!(X, X * USVᴴ[3]', USVᴴ[1]')
        diagview(USVᴴ[2]) .= sqrt.(diagview(USVᴴ[2]))
        UsqrtS = rmul!(USVᴴ[1], USVᴴ[2])
        mul!(Y, UsqrtS, UsqrtS')
    end
    logU = project_antihermitian!(log(U))
    if eltype(U) <: Real
        @assert mapreduce(abs ∘ imag, max, logU; init=abs(zero(eltype(logU)))) <= tol
        K = real(logU)
    else
        K = logU
    end
    C = view(K, (p + 1):r, (p + 1):r)
    i = 1
    τ = mapreduce(abs, max, C; init=abs(zero(eltype(C))))
    while τ > tol
        if i > maxiter
            @warn "Stiefel logarithm: not converged in $maxiter iterations, τ = $τ"
            break
        end
        eC = exp(rmul!(C, -1))
        X .= X * eC
        Y .= Y * eC
        logU = project_antihermitian!(log(U))
        if eltype(U) <: Real
            @assert mapreduce(abs ∘ imag, max, logU; init=abs(zero(eltype(logU)))) <= tol
            K .= real.(logU)
        else
            K .= logU
        end
        τ = maximum(abs, C)
        i += 1
    end
    return K[1:p, 1:p], Q, K[(p + 1):r, 1:p]
end
