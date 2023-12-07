include("../main.jl")
using JLD2
using ProgressMeter

## Parameters
max_level = 50
levels = 0:max_level

fock_basis = Iterators.product(levels, levels) |> collect |> permutedims |> vec

λs = [1]
σs = [1 / 2]
params = [(λ, σ) for λ in λs, σ in σs] |> vec

println("Computing 2D")
for p in params
    λ = p[1]
    σ = p[2]

    if !isfile("data/2D/Φ_mat_2D_λ$(λ)_σ$(σ)_levels$(max_level).jld2")

        Φ_mat = zeros(ComplexF64, length(fock_basis), length(fock_basis))
        pr = Progress(length(fock_basis))

        for ii in eachindex(fock_basis)

            r = zeros(ComplexF64, ii)

            Threads.@threads for jj = 1:ii
                x = fock_basis[ii]
                y = fock_basis[jj]
                r[jj] = Σ(1 / (2 * σ^2), x[1], y[1]) * Σ(λ^2 / (2 * σ^2), x[2], y[2])
                GC.safepoint()
            end
            Φ_mat[ii, 1:ii] = r
            Φ_mat[1:ii, ii] = r
            next!(pr)
        end

        save_object(
            "data/2D/Φ_mat_2D_λ$(λ)_σ$(σ)_levels$(max_level).jld2",
            (fock_basis, Φ_mat),
        )
    end

end

println("Computing 2Osc")
for p in params
    λ = p[1]
    σ = p[2]

    if !isfile("data/2Osc/Φ_mat_2Osc_λ$(λ)_σ$(σ)_levels$(max_level).jld2")

        # Fourier transform of the interaction
        function Φ(q)
            return (exp(-q^2 * σ^2 / 2) * σ / √(2 * π))
        end

        Φ_mat = zeros(ComplexF64, length(fock_basis), length(fock_basis))
        pr = Progress(length(fock_basis))

        for ii in eachindex(fock_basis)

            r = zeros(ComplexF64, ii)

            Threads.@threads for jj = 1:ii
                x = fock_basis[ii]
                y = fock_basis[jj]
                r[jj] = Φ_element_2Osc(Φ, λ, x, y)[1]
                GC.safepoint()
            end

            Φ_mat[ii, 1:ii] = r
            Φ_mat[1:ii, ii] = r
            next!(pr)
        end

        save_object(
            "data/2Osc/Φ_mat_2Osc_λ$(λ)_σ$(σ)_levels$(max_level).jld2",
            (fock_basis, Φ_mat),
        )
    end

end
