module TensorKitManifolds

export base, checkbase, isisometry, isunitary
export projectcomplement, projectcomplement!
export Grassmann, Stiefel, Unitary
export inner, retract, transport, transport!

using TensorKit
using MatrixAlgebraKit: MatrixAlgebraKit, AbstractAlgorithm, Algorithm, PolarViaSVD,
    LAPACK_DivideAndConquer, diagview
import MatrixAlgebraKit as MAK

using Base: @deprecate

# Every submodule -- Grassmann, Stiefel, and Unitary -- implements their own methods for
# these. The signatures should be
# inner(W, Δ₁::Tangent, Δ₂::Tangent; metric)
# retract(W, Δ::Tangent, α::Real; alg)
# transport(Θ::Tangent, W, Δ::Tangent, α::Real, W′; alg)
# where the keyword arguments `alg` and `metric` should always be accepted, even if there is
# only one option for them and they are ignored. The `Tangent` is just a placeholder for the
# tangent type of each manifold. Similarly each submodule defines a `project!` function,
# which too should accept a keyword argument `metric`, even if it is ignored.
function inner end
function retract end
function transport end
function transport! end
function base end
function checkbase end
checkbase(x, y, z, args...) = checkbase(checkbase(x, y), z, args...)

# the machine epsilon for the elements of an object X, name inspired from eltype
scalareps(X) = eps(real(scalartype(X)))

@deprecate projecthermitian(W) MAK.project_hermitian(W)
@deprecate projecthermitian!(W) MAK.project_hermitian!(W)

@deprecate projectantihermitian(W) MAK.project_antihermitian(W)
@deprecate projectantihermitian!(W) MAK.project_antihermitian!(W)

@deprecate projectisometric(W; kwargs...) MAK.project_isometric(W; kwargs...)
@deprecate projectisometric!(W; kwargs...) MAK.project_isometric!(W; kwargs...)

function projectcomplement(X::AbstractTensorMap, W::AbstractTensorMap, kwargs...)
    return projectcomplement!(copy(X), W; kwargs...)
end
function projectcomplement!(
        X::AbstractTensorMap, W::AbstractTensorMap;
        tol = 10 * scalareps(X)
    )
    P = W' * X
    nP = norm(P)
    nX = norm(X)
    dP = dim(P)
    while nP > tol * max(dP, nX)
        X = mul!(X, W, P, -1, 1)
        P = W' * X
        nP = norm(P)
    end
    return X
end

include("auxiliary.jl")
include("grassmann.jl")
include("stiefel.jl")
include("unitary.jl")

end
