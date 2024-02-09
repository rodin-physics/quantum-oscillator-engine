using HypergeometricFunctions
using LinearAlgebra
using Polynomials
using QuadGK
using SparseArrays
using SpecialFunctions
using SpecialPolynomials

dirs = [
    "data",
    "data/2Osc",
    "data/2D",
    "data/3D_Osc_Engine/",
    "data/3D_Osc_Engine/animation",
    "data/3D_Osc_Engine/compression_expansion",
    "data/3D_Osc_Engine/benchmarking",
    "data/3D_Osc_Engine/engine_operation",
    "data/3D_Osc_Engine/thermal_contact/",
]
[isdir(d) ? nothing : mkdir(d) for d in dirs]

## GENERAL
# Harmonic oscillator wave function
# function Ψ(x, n, l)
#     res =
#         basis(Hermite, n)(x / l) * exp(-x^2 / 2 / l^2) / √(2^n * factorial(big(n))) / √(l) /
#         π^(1 / 4)
#     return res
# end

function Ψ(x, n, l)
    res =
        basis(Hermite, n)(x / l) * exp(-x^2 / 2 / l^2 - log(2) * n / 2) /
        √(factorial(big(n))) / √(l) / π^(1 / 4)
    return res
end

function real_space(Ψ_Fock, fock_basis::Vector{Tuple{Int,Int}}, xs::Vector{Float64}, λ)
    r = zeros(ComplexF64, length(xs), length(xs))
    for ii in eachindex(fock_basis)
        Ψ1 = [Ψ(x, fock_basis[ii][1], 1) for x in xs]
        Ψ2 = [Ψ(x, fock_basis[ii][2], λ) for x in xs]
        r = r + Ψ_Fock[ii] .* Ψ1 * permutedims(Ψ2)
    end
    return r
end

function fock_space(Ψ_real, l::Float64, max_fock::Int)
    cs = zeros(max_fock + 1)
    for ii in eachindex(cs)
        c = quadgk(x -> Ψ_real(x) * Ψ(x, ii - 1, l), -Inf, Inf, atol = 1e-3)
        cs[ii] = c[1]
    end
    return (cs)
end

function partial_trace(ρ, n)
    # 'n' tells which state is being traced out
    s = size(ρ)[1] |> sqrt |> Int
    ρ_tr = zeros(ComplexF64, s, s)
    for ii = 0:(s-1)
        idx = findall(
            x -> x[n] == ii,
            Iterators.product(0:(s-1), 0:(s-1)) |> collect |> permutedims |> vec,
        )
        idx = Iterators.product(idx, idx) |> collect
        ρ_tr = ρ_tr + [ρ[ii...] for ii in idx]
    end
    return ρ_tr
end

function unitary_evolution(H, τ)
    return exp(-2im * π * H * τ)
end

function RKstep(H, current_state, t, δτ)

    function derivative(τ, state)
        return (-2im * π .* H(τ) * state)
    end

    t1 = t
    t2 = t + δτ / 4
    t3 = t + δτ / 4
    t4 = t + δτ / 2
    t5 = t + 3 * δτ / 4
    t6 = t + δτ

    k1 = derivative(t1, current_state)
    k2 = derivative(t2, current_state .+ k1 .* (δτ / 4))
    k3 = derivative(t3, current_state .+ (k1 .+ k2) .* (δτ / 8))
    k4 = derivative(t4, current_state .+ k3 .* δτ .- k2 .* (δτ / 2))
    k5 = derivative(t5, current_state .+ k1 .* (δτ * 3 / 16) .+ k4 .* (δτ * 9 / 16))
    k6 = derivative(
        t6,
        current_state .- k1 .* (3 / 7 * δτ) .+ k2 .* (2 / 7 * δτ) .+ k3 .* (12 / 7 * δτ) .- k4 .* (12 / 7 * δτ) .+ k5 .* (8 / 7 * δτ),
    )
    res =
        current_state .+
        (δτ / 90) .* (7 .* k1 .+ 32 .* k3 .+ 12 .* k4 .+ 32 .* k5 .+ 7 .* k6)
    # res = res ./ tr(res)
    return res
end

function trace_distance(ρ1, ρ2)
    res = (sqrt((ρ1 - ρ2) * (ρ1 - ρ2)') |> tr) / 2
    return res
end

## MEASUREMENT ENGINE
function Ξ(x, u, j)
    m = min(u, j)
    M = max(u, j)
    a = abs(j - u)
    L_idx = zeros(Int, m + 1)
    L_idx[end] = 1
    L = Laguerre{a}(L_idx)

    xi =
        exp(-(x^2) / 4) *
        (x / 1im / √(2))^a *
        (√(factorial(big(m)) / factorial(big(M)))) *
        L(x^2 / 2)

    return (xi)
end

function Σ(x, u, j)
    if isodd(u + j)
        return 0
    else
        mn = (u + j) / 2
        sigma =
            factorial(big(u + j)) / factorial(big(mn)) /
            √(factorial(big(u)) * factorial(big(j))) *
            (1 / (1 + x)^(mn + 1 / 2)) *
            (-x / 2)^mn *
            HypergeometricFunctions._₂F₁general(-u, -j, 1 / 2 - mn, (1 + x) / (2 * x))

        return sigma
    end
end

function Φ_element_2Osc(Φ, λ, (u, v), (j, k))
    f(x) = Φ(x) * Ξ(x, u, j) * Ξ(-x * λ, v, k)
    r = quadgk(f, -Inf, Inf, rtol = 1e-4)
    return r
end

## OTTO ENGINE
function mode_interaction(n, m, l, d, σ)
    res = quadgk(
        x -> Ψ(x, n, l) * Ψ(x, m, l) * exp(-(x - d)^2 / (2 * σ^2)),
        -Inf,
        Inf,
        atol = 1e-5,
    )[1]
    return res
end

## PISTON ENGINE
function compression_interaction(n, m, l, p)
    res = quadgk(x -> Ψ(x, n, l) * Ψ(x, m, l), -Inf, p, atol = 1e-5)[1]
    return res
end

# function overlap(n, m, λ)
#     if isodd(n + m)
#         return 0.0
#     else
#         r = quadgk(
#             x -> isnan(Ψ(x, n, λ) * Ψ(x, m, 1)) ? 0 : (Ψ(x, n, λ) * Ψ(x, m, 1)),
#             -Inf,
#             Inf,
#             atol = 1e-5,
#             rtol = 1e-4,
#         )
#         return Float64(r[1])
#     end
# end

# function overlap(n, m, λ)
#     if isodd(n + m)
#         return 0.0
#     else
#         r = quadgk(x -> Ψ(x, n, λ) * Ψ(x, m, 1), -Inf, Inf, atol = 1e-5, rtol = 1e-4)
#         return Float64(r[1])
#     end
# end
# overlap(0, 100, 1 / sqrt(5))

# r = [(Ψ(x, 70, 1)) for x = -1:0.01:1]
# n = 62
# 1 / √(big(2^n) * factorial(big(n)))

# function Ψ1(x, n, l)
#     res =
#         basis(Hermite, n)(x / l) * exp(-x^2 / 2 / l^2 - log(2) * n / 2) /
#         √(factorial(big(n))) / √(l) / π^(1 / 4)
#     return res
# end

# Ψ(2, 62, 1) ≈ Ψ1(2, 62, 1)
# Float64(Ψ(2, 62, 1)) / Float64(Ψ1(2, 62, 1))
