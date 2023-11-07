using HypergeometricFunctions
using LinearAlgebra
using Polynomials
using QuadGK
using SparseArrays
using SpecialFunctions
using SpecialPolynomials

dirs = ["data", "data/2Osc", "data/2D"]
[isdir(d) ? nothing : mkdir(d) for d in dirs]

# Harmonic oscillator wave function
function Ψ(x, n, l)
    res =
        basis(Hermite, n)(x / l) * exp(-x^2 / 2 / l^2) / √(2^n * factorial(big(n))) / √(l) /
        π^(1 / 4)
    return res
end

# Auxiliary functions
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

function unitary_evolution(H, τ)
    return exp(-2im * π * H * τ)
end
