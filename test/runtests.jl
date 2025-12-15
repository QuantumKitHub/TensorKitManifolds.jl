using TensorKit, TensorKitManifolds
using VectorInterface
using Test, TestExtras

spaces = (
    ℂ^4, ℤ₂Space(2, 2), U₁Space(0 => 2, 1 => 1, -1 => 1),
    SU₂Space(0 => 2, 1 / 2 => 1),
)
const ϵ = 1.0e-7
const α = 0.75

@testset "Grassmann with space $V" for V in spaces
    for T in (Float64, ComplexF64)
        W, = left_polar(randn(T, V * V * V, V * V))
        X = randn(T, space(W))
        Y = randn(T, space(W))
        Δ = @constinferred Grassmann.project(X, W)
        Θ = Grassmann.project(Y, W)
        γ = randn(T)
        Ξ = -Δ + γ * Θ
        @test γ * Θ ≈ scale(Θ, γ)
        @test γ * Θ ≈ scale!(copy(Θ), γ)
        @test γ * Θ ≈ scale!!(copy(Θ), γ)
        @test γ * Θ ≈ scale!(copy(Θ), Θ, γ)
        @test γ * Θ ≈ scale!!(copy(Θ), Θ, γ)
        @test Ξ ≈ add(Δ, Θ, γ, -1)
        @test Ξ ≈ add!(copy(Δ), Θ, γ, -1)
        @test Ξ ≈ add!!(copy(Δ), Θ, γ, -1)
        @test norm(W' * Δ[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W' * Θ[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W' * Ξ[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(zero(W)) == 0
        @test (@constinferred Grassmann.inner(W, Δ, Θ)) ≈ real(inner(Δ[], Θ[]))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(inner(X, Θ[]))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(inner(Δ[], Y))
        @test Grassmann.inner(W, Δ, Δ) ≈ norm(Δ[])^2

        W2, = @constinferred Grassmann.retract(W, Δ, ϵ)
        @test W2 ≈ W + ϵ * Δ[]
        W2, Δ2′ = Grassmann.retract(W, Δ, α)
        @test norm(W2' * Δ2′[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test Δ2′[] ≈
            (
            first(Grassmann.retract(W, Δ, α + ϵ / 2)) -
                first(Grassmann.retract(W, Δ, α - ϵ / 2))
        ) / (ϵ) atol = dim(W) * ϵ
        Δ2 = @constinferred Grassmann.transport(Δ, W, Δ, α, W2)
        Θ2 = Grassmann.transport(Θ, W, Δ, α, W2)
        Ξ2 = Grassmann.transport(Ξ, W, Δ, α, W2)
        @test Δ2[] ≈ Δ2′[]
        @test norm(W2' * Δ2[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W2' * Θ2[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W2' * Ξ2[]) <= sqrt(eps(real(T))) * dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Grassmann.inner(W2, Δ2, Θ2) ≈ Grassmann.inner(W, Δ, Θ)
        @test Grassmann.inner(W2, Ξ2, Θ2) ≈ Grassmann.inner(W, Ξ, Θ)

        Wend = randisometry(T, codomain(W), domain(W))
        Δ3, V1 = Grassmann.invretract(W, Wend)
        @test Wend ≈ Grassmann.retract(W, Δ3, 1)[1] * V1
        U = Grassmann.relativegauge(W, Wend)
        V2 = Grassmann.invretract(W, Wend * U)[2]
        @test V2 ≈ one(V2)
    end
end

@testset "Stiefel with space $V" for V in spaces
    V1 = V * V
    V2 = fuse(V * V * V) ⊖ fuse(V1)
    for T in (Float64, ComplexF64)
        for W in (randisometry(T, V * V * V, V1), randisometry(T, V * V * V, V2))
            X = randn(T, space(W))
            Y = randn(T, space(W))
            Δ = @constinferred Stiefel.project_euclidean(X, W)
            Θ = Stiefel.project_canonical(Y, W)
            γ = rand()
            Ξ = -Δ + γ * Θ
            @test γ * Θ ≈ scale(Θ, γ)
            @test γ * Θ ≈ scale!(copy(Θ), γ)
            @test γ * Θ ≈ scale!!(copy(Θ), γ)
            @test γ * Θ ≈ scale!(copy(Θ), Θ, γ)
            @test γ * Θ ≈ scale!!(copy(Θ), Θ, γ)
            @test Ξ ≈ add(Δ, Θ, γ, -1)
            @test Ξ ≈ add!(copy(Δ), Θ, γ, -1)
            @test Ξ ≈ add!!(copy(Δ), Θ, γ, -1)
            @test norm(W' * Δ[] + Δ[]' * W) <= sqrt(eps(real(T))) * dim(domain(W))
            @test norm(W' * Θ[] + Θ[]' * W) <= sqrt(eps(real(T))) * dim(domain(W))
            @test norm(W' * Ξ[] + Ξ[]' * W) <= sqrt(eps(real(T))) * dim(domain(W))
            @test (@constinferred Stiefel.inner_euclidean(W, Δ, Θ)) ≈ real(inner(Δ[], Θ[]))
            @test (@constinferred Stiefel.inner_canonical(W, Δ, Θ)) ≈
                real(inner(Δ[], Θ[] - W * (W' * Θ[]) / 2))
            @test Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(inner(X, Θ[]))
            @test !(Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(inner(Δ[], Y)))
            @test !(Stiefel.inner_canonical(W, Δ, Θ) ≈ real(inner(X, Θ[])))
            @test Stiefel.inner_canonical(W, Δ, Θ) ≈ real(inner(Δ[], Y))
            @test Stiefel.inner_euclidean(W, Δ, Δ) ≈ norm(Δ[])^2
            @test Stiefel.inner_canonical(W, Δ, Δ) ≈
                (1 // 2) * norm(W' * Δ[])^2 + norm(Δ[] - W * (W'Δ[]))^2

            W2, = @constinferred Stiefel.retract_exp(W, Δ, ϵ)
            @test W2 ≈ W + ϵ * Δ[]
            W2, Δ2′ = Stiefel.retract_exp(W, Δ, α)
            @test norm(W2' * Δ2′[] + Δ2′[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test Δ2′[] ≈
                (
                first(Stiefel.retract_exp(W, Δ, α + ϵ / 2)) -
                    first(Stiefel.retract_exp(W, Δ, α - ϵ / 2))
            ) / (ϵ) atol = dim(W) * ϵ
            Δ2 = @constinferred Stiefel.transport_exp(Δ, W, Δ, α, W2)
            Θ2 = Stiefel.transport_exp(Θ, W, Δ, α, W2)
            Ξ2 = Stiefel.transport_exp(Ξ, W, Δ, α, W2)
            @test Δ2′[] ≈ Δ2[]
            @test norm(W2' * Δ2[] + Δ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test norm(W2' * Θ2[] + Θ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test norm(W2' * Ξ2[] + Ξ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
            @test Stiefel.inner_euclidean(W2, Δ2, Θ2) ≈ Stiefel.inner_euclidean(W, Δ, Θ)
            @test Stiefel.inner_euclidean(W2, Ξ2, Θ2) ≈ Stiefel.inner_euclidean(W, Ξ, Θ)
            @test Stiefel.inner_canonical(W2, Δ2, Θ2) ≈ Stiefel.inner_canonical(W, Δ, Θ)
            @test Stiefel.inner_canonical(W2, Ξ2, Θ2) ≈ Stiefel.inner_canonical(W, Ξ, Θ)

            W2, = @constinferred Stiefel.retract_cayley(W, Δ, ϵ)
            @test W2 ≈ W + ϵ * Δ[]
            W2, Δ2′ = Stiefel.retract_cayley(W, Δ, α)
            @test norm(W2' * Δ2′[] + Δ2′[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test Δ2′[] ≈
                (
                first(Stiefel.retract_cayley(W, Δ, α + ϵ / 2)) -
                    first(Stiefel.retract_cayley(W, Δ, α - ϵ / 2))
            ) / (ϵ) atol = dim(W) * ϵ
            @test norm(Δ2′) <= norm(Δ)
            Δ2 = @constinferred Stiefel.transport_cayley(Δ, W, Δ, α, W2)
            Θ2 = Stiefel.transport_cayley(Θ, W, Δ, α, W2)
            Ξ2 = Stiefel.transport_cayley(Ξ, W, Δ, α, W2)
            @test !(Δ2′[] ≈ Δ2[])
            @test norm(W2' * Δ2[] + Δ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test norm(W2' * Θ2[] + Θ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test norm(W2' * Ξ2[] + Ξ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
            @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
            @test Stiefel.inner_euclidean(W2, Δ2, Θ2) ≈ Stiefel.inner_euclidean(W, Δ, Θ)
            @test Stiefel.inner_euclidean(W2, Ξ2, Θ2) ≈ Stiefel.inner_euclidean(W, Ξ, Θ)
            @test Stiefel.inner_canonical(W2, Δ2, Θ2) ≈ Stiefel.inner_canonical(W, Δ, Θ)
            @test Stiefel.inner_canonical(W2, Ξ2, Θ2) ≈ Stiefel.inner_canonical(W, Ξ, Θ)

            W3 = project_isometric!(W + 1.0e-1 * rand(T, codomain(W), domain(W)))
            Δ3 = Stiefel.invretract(W, W3)
            @test W3 ≈ Stiefel.retract(W, Δ3, 1)[1]
        end
    end
end

@testset "Unitary with space $V" for V in spaces
    for T in (Float64, ComplexF64)
        W, = left_polar(randn(T, V * V * V, V * V))
        X = randn(T, space(W))
        Y = randn(T, space(W))
        Δ = @constinferred Unitary.project(X, W)
        Θ = Unitary.project(Y, W)
        γ = randn()
        Ξ = -Δ + γ * Θ
        @test γ * Θ ≈ scale(Θ, γ)
        @test γ * Θ ≈ scale!(copy(Θ), γ)
        @test γ * Θ ≈ scale!!(copy(Θ), γ)
        @test γ * Θ ≈ scale!(copy(Θ), Θ, γ)
        @test γ * Θ ≈ scale!!(copy(Θ), Θ, γ)
        @test Ξ ≈ add(Δ, Θ, γ, -1)
        @test Ξ ≈ add!(copy(Δ), Θ, γ, -1)
        @test Ξ ≈ add!!(copy(Δ), Θ, γ, -1)
        @test norm(W' * Δ[] + Δ[]' * W) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W' * Θ[] + Θ[]' * W) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W' * Ξ[] + Ξ[]' * W) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(zero(W)) == 0
        @test (@constinferred Unitary.inner(W, Δ, Θ)) ≈ real(inner(Δ[], Θ[]))
        @test Unitary.inner(W, Δ, Θ) ≈ real(inner(X, Θ[]))
        @test Unitary.inner(W, Δ, Θ) ≈ real(inner(Δ[], Y))
        @test Unitary.inner(W, Δ, Δ) ≈ norm(Δ[])^2

        W2, = @constinferred Unitary.retract(W, Δ, ϵ)
        @test W2 ≈ W + ϵ * Δ[]
        W2, Δ2′ = Unitary.retract(W, Δ, α)
        @test norm(W2' * Δ2′[] + Δ2′[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
        @test Δ2′[] ≈
            (
            first(Unitary.retract(W, Δ, α + ϵ / 2)) -
                first(Unitary.retract(W, Δ, α - ϵ / 2))
        ) / (ϵ) atol = dim(W) * ϵ

        Δ2 = @constinferred Unitary.transport_parallel(Δ, W, Δ, α, W2)
        Θ2 = Unitary.transport_parallel(Θ, W, Δ, α, W2)
        Ξ2 = Unitary.transport_parallel(Ξ, W, Δ, α, W2)
        @test Δ2′[] ≈ Δ2[]
        @test norm(W2' * Θ2[] + Θ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W2' * Ξ2[] + Ξ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Unitary.inner(W2, Δ2, Θ2) ≈ Unitary.inner(W, Δ, Θ)
        @test Unitary.inner(W2, Ξ2, Θ2) ≈ Unitary.inner(W, Ξ, Θ)

        Δ2 = @constinferred Unitary.transport_stiefel(Δ, W, Δ, α, W2)
        Θ2 = Unitary.transport_stiefel(Θ, W, Δ, α, W2)
        Ξ2 = Unitary.transport_stiefel(Ξ, W, Δ, α, W2)
        @test Δ2′[] ≈ Δ2[]
        @test norm(W2' * Δ2[] + Δ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W2' * Θ2[] + Θ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
        @test norm(W2' * Ξ2[] + Ξ2[]' * W2) <= sqrt(eps(real(T))) * dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Unitary.inner(W2, Δ2, Θ2) ≈ Unitary.inner(W, Δ, Θ)
        @test Unitary.inner(W2, Ξ2, Θ2) ≈ Unitary.inner(W, Ξ, Θ)
    end
end

using Aqua
Aqua.test_all(TensorKitManifolds)
