module Unitary

# unitary U
# tangent vectors Δ = U*A with A' = -A

using TensorKit
import TensorKit: similarstoragetype, SectorDict
import ..TensorKitManifolds: base, checkbase, inner, retract, transport, transport!
import MatrixAlgebraKit as MAK
import VectorInterface as VI

struct UnitaryTangent{T <: AbstractTensorMap, TA <: AbstractTensorMap}
    W::T
    A::TA
    function UnitaryTangent(
            W::AbstractTensorMap{T₁, S, N₁, N₂},
            A::AbstractTensorMap{T₂, S, N₂, N₂}
        ) where {T₁, T₂, S, N₁, N₂}
        T = typeof(W)
        TA = typeof(A)
        return new{T, TA}(W, A)
    end
end
Base.copy(Δ::UnitaryTangent) = UnitaryTangent(Δ.W, copy(Δ.A))
Base.getindex(Δ::UnitaryTangent) = Δ.W * Δ.A
base(Δ::UnitaryTangent) = Δ.W
function checkbase(Δ₁::UnitaryTangent, Δ₂::UnitaryTangent)
    return Δ₁.W == Δ₂.W ? Δ₁.W :
        throw(ArgumentError("tangent vectors with different base points"))
end

# Basic vector space behaviour
function Base.:+(Δ₁::UnitaryTangent, Δ₂::UnitaryTangent)
    return UnitaryTangent(checkbase(Δ₁, Δ₂), Δ₁.A + Δ₂.A)
end
function Base.:-(Δ₁::UnitaryTangent, Δ₂::UnitaryTangent)
    return UnitaryTangent(checkbase(Δ₁, Δ₂), Δ₁.A - Δ₂.A)
end
Base.:-(Δ::UnitaryTangent) = (-1) * Δ

Base.:*(Δ::UnitaryTangent, α::Real) = UnitaryTangent(base(Δ), Δ.A * α)
Base.:*(α::Real, Δ::UnitaryTangent) = UnitaryTangent(base(Δ), α * Δ.A)
Base.:/(Δ::UnitaryTangent, α::Real) = UnitaryTangent(base(Δ), Δ.A / α)
Base.:\(α::Real, Δ::UnitaryTangent) = UnitaryTangent(base(Δ), α \ Δ.A)

Base.zero(Δ::UnitaryTangent) = UnitaryTangent(Δ.W, zero(Δ.A))

function Base.isapprox(Δ₁::UnitaryTangent, Δ₂::UnitaryTangent; kwargs...)
    checkbase(Δ₁, Δ₂)
    return isapprox(Δ₁.A, Δ₂.A; kwargs...)
end

# VectorInterface methods
VI.scalartype(Δ::UnitaryTangent) = VI.scalartype(Δ.A)

function VI.zerovector(Δ::UnitaryTangent, T::Type{<:Number} = VI.scalartype(Δ))
    return UnitaryTangent(base(Δ), VI.zerovector(Δ.A, T))
end
function VI.zerovector!(Δ::UnitaryTangent)
    VI.zerovector!(Δ.A)
    return Δ
end
VI.zerovector!!(Δ::UnitaryTangent) = VI.zerovector!(Δ)

function VI.scale(Δ::UnitaryTangent, α::Real)
    return UnitaryTangent(base(Δ), VI.scale(Δ.A, α))
end
function VI.scale!(Δ::UnitaryTangent, α::Real)
    VI.scale!(Δ.A, α)
    return Δ
end
function VI.scale!!(Δ::UnitaryTangent, α::Real)
    A′ = VI.scale!!(Δ.A, α)
    return A′ === Δ.A ? Δ : UnitaryTangent(base(Δ), A′)
end

function VI.scale!(Δy::UnitaryTangent, Δx::UnitaryTangent, α::Real)
    VI.scale!(Δy.A, Δx.A, α)
    return Δy
end
function VI.scale!!(Δy::UnitaryTangent, Δx::UnitaryTangent, α::Real)
    A′ = VI.scale!!(Δy.A, Δx.A, α)
    return A′ === Δy.A ? Δy : UnitaryTangent(base(Δy), A′)
end

function VI.add(Δy::UnitaryTangent, Δx::UnitaryTangent, α::Real, β::Real)
    return UnitaryTangent(checkbase(Δy, Δx), VI.add(Δy.A, Δx.A, α, β))
end
function VI.add!(Δy::UnitaryTangent, Δx::UnitaryTangent, α::Real, β::Real)
    checkbase(Δy, Δx)
    VI.add!(Δy.A, Δx.A, α, β)
    return Δy
end
function VI.add!!(Δy::UnitaryTangent, Δx::UnitaryTangent, α::Real, β::Real)
    checkbase(Δy, Δx)
    A′ = VI.add!!(Δy.A, Δx.A, α, β)
    return A′ === Δy.A ? Δy : UnitaryTangent(base(Δy), A′)
end

function VI.inner(Δ₁::UnitaryTangent, Δ₂::UnitaryTangent)
    checkbase(Δ₁, Δ₂)
    return VI.inner(Δ₁.A, Δ₂.A)
end

VI.norm(Δ::UnitaryTangent, p::Real = 2) = norm(Δ.A, p)

# For backward compatibility: LinearAlgebra methods
TensorKit.rmul!(Δ::UnitaryTangent, α::Real) = VI.scale!(Δ, α)
TensorKit.lmul!(α::Real, Δ::UnitaryTangent) = VI.scale!(Δ, α)
TensorKit.axpy!(α::Real, Δx::UnitaryTangent, Δy::UnitaryTangent) = VI.add!(Δy, Δx, α)
TensorKit.axpby!(α::Real, Δx::UnitaryTangent, β::Real, Δy::UnitaryTangent) = VI.add!(Δy, Δx, α, β)
TensorKit.dot(Δ₁::UnitaryTangent, Δ₂::UnitaryTangent) = VI.inner(Δ₁, Δ₂)

# tangent space methods
function inner(
        W::AbstractTensorMap, Δ₁::UnitaryTangent, Δ₂::UnitaryTangent;
        metric = :euclidean
    )
    @assert metric == :euclidean
    return Δ₁ === Δ₂ ? norm(Δ₁)^2 : real(dot(Δ₁, Δ₂))
end
function project!(X::AbstractTensorMap, W::AbstractTensorMap; metric = :euclidean)
    @assert metric == :euclidean
    P = W' * X
    A = project_antihermitian!(P)
    return UnitaryTangent(W, A)
end
project(X, W; metric = :euclidean) = project!(copy(X), W; metric = :euclidean)

# geodesic retraction, coincides with Stiefel retraction (which is not geodesic for p < n)
function retract(W::AbstractTensorMap, Δ::UnitaryTangent, α; alg = MAK.select_algorithm(left_polar!, W))
    W == base(Δ) || throw(ArgumentError("not a valid tangent vector at base point"))
    E = exp(α * Δ.A)
    W′ = project_isometric!(W * E; alg)
    A′ = Δ.A
    return W′, UnitaryTangent(W′, A′)
end

# isometric vector transports compatible with above retraction
# (also with differential of retraction)
function transport!(Θ::UnitaryTangent, W::AbstractTensorMap, Δ::UnitaryTangent, α::Real, W′; alg = :stiefel)
    if alg == :parallel
        return transport_parallel!(Θ, W, Δ, α, W′)
    elseif alg == :stiefel
        return transport_stiefel!(Θ, W, Δ, α, W′)
    else
        throw(ArgumentError("unknown algorithm: `alg = $metric`"))
    end
end
function transport(Θ::UnitaryTangent, W::AbstractTensorMap, Δ::UnitaryTangent, α::Real, W′; alg = :stiefel)
    return transport!(copy(Θ), W, Δ, α, W′; alg)
end

# transport_parallel correspondings to the torsion-free Levi-Civita connection
# transport_stiefel is compatible to Stiefel.transport and corresponds to a non-torsion-free
# connection
function transport_parallel!(Θ::UnitaryTangent, W::AbstractTensorMap, Δ::UnitaryTangent, α, W′)
    W == checkbase(Δ, Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    E = exp((α / 2) * Δ.A)
    A′ = project_antihermitian!(E' * Θ.A * E) # exra projection for stability
    return UnitaryTangent(W′, A′)
end
function transport_parallel(Θ::UnitaryTangent, W::AbstractTensorMap, Δ::UnitaryTangent, α, W′)
    return transport_parallel!(copy(Θ), W, Δ, α, W′)
end

function transport_stiefel!(Θ::UnitaryTangent, W::AbstractTensorMap, Δ::UnitaryTangent, α, W′)
    W == checkbase(Δ, Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    A′ = Θ.A
    return UnitaryTangent(W′, A′)
end
function transport_stiefel(Θ::UnitaryTangent, W::AbstractTensorMap, Δ::UnitaryTangent, α, W′)
    return transport_stiefel!(copy(Θ), W, Δ, α, W′)
end

end
