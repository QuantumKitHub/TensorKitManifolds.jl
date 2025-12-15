# TensorKitManifolds [![Build Status](https://github.com/Jutho/TensorKitManifolds.jl/workflows/CI/badge.svg)](https://github.com/Jutho/TensorKitManifolds.jl/actions) [![Coverage](https://codecov.io/gh/Jutho/TensorKitManifolds.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Jutho/TensorKitManifolds.jl)

# What is new in v0.8

* TensorKitManifolds.jl now provides VectorInterface.jl compatibility for the tangent
  vectors on the manifolds.
* TensorKitManifolds.jl no longer exports any methods, and only exports the three manifold
  types. All methods now need to be quantified as e.g. `Stiefel.project!`, `Grassmann.inner`
  etc, or explicitly imported.
* There is a difference between `VectorInterface.inner`, which computes the complex inner
  product between two tangent vectors as vectors in a complex Euclidean space, and the
  `inner` method in TensorKitManifolds.jl or its submodules, which computes the Riemannian
  inner product (metric) on the manifold and is always real-valued.

# Overview

There are three manifolds: `Grassmann`, `Stiefel` and `Unitary`, corresponding to submodules
of TensorKitManifolds, whose names are exported.

These modules have a number of public but non-exported methods, namely:
* `Δ = Grassmann.project(!)(X, W; metric = :euclidean)`: project an arbitrary tensor `X`
  onto the tangent space of the manifold at the point `W`, returned as a tangent vector `Δ`
  of a specific type, that also stores the base point `W`. The exclamation mark denotes that
  `X` is destroyed in the process.
* `W = Grassmann.base(Δ)`: return the base point `W` of the tangent vector `Δ`.
* `s = Grassmann.inner(W, Δ₁, Δ₂; metric = :euclidean)`: Riemannian inner product (a.k.a.
  metric) between tangent vectors `Δ₁` and `Δ₂` at the point `W`, which is a real-valued
  scalar `s`. For the default `:euclidean` metric, this is equal to the real part of the
  complex inner product between `Δ₁` and `Δ₂` as vectors in a complex Euclidean space, but
  other metrics might also be available.
* ` W′, Δ′ = Grassmann.retract(W, Δ, α)`: retract the point `W` in the direction of tangent
  vector `Δ` with step length `α`, ending up in the point `W′`, and return also the local
  directional derivative along the path `Δ′`, which is a tangent vector at `W′`.
* `Θ′ = Grassmann.transport(!)(Θ, W, Δ, α, W′)`: transport tangent vector `Θ` along the
  retraction of `W` in the direction of `Δ` with step length `α`, which ends at `W′`. The
  resulting transported vector `Θ′` is a tangent vector with base point `W′`. The method
  with exclamation mark destroys `Θ` in the process.

The same methods exist for `Stiefel` and `Unitary` manifolds. When multiple implementations
or metrics are avaible, they are specified using a keyword argument to the above methods, or
explicitly as `Stiefel.inner_euclidean`, `Stiefel.inner_canonical`,
`Stiefel.project_euclidean(!)`, `Stiefel.project_canonical(!)`, `Stiefel.retract_exp`,
`Stiefel.transport_exp(!)`, `Stiefel.retract_cayley`, `Stiefel.transport_cayley(!)`,
`Unitary.transport_parallel(!)`, `Unitary.transport_stiefel(!)`.




 have a function `Δ = project(!)(X,W)` (e.g. `Grassmann.project(!)` etc) to project an
 arbitrary tensor `X` onto the tangent space of `W`, which is assumed to be
 isometric/unitary (not checked). The exclamation mark denotes that `X` is destroyed in the
 process. The result `Δ` is of a specific type, the corresponding `TensorMap` object can be
 obtained via an argumentless `getindex`, i.e. `Δ[]` returns the corresponding `TensorMap`.
 However, you typically don't need those. The base point `W` is also stored in `Δ` and can
 be returned using `W = base(Δ)`. Hence, `Δ` should be assumed to be a point `(W, Δ[])` on
 the tangent bundle of the manifold.

The objects `Δ` returned by `project(!)` also satisfy the behaviour of vector: they have
scalar multiplication, addition, left and right in-place multiplication with scalars using
`lmul!` and `rmul!`, `axpy!` and `axpby!` as well as complex euclidean inner product `dot`
and corresponding `norm`. When combining two tangent vectors using addition or inner
product, they need to have the same `base`.


