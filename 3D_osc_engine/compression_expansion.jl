include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

## Parameters
Ts = [1, 5]                 # Temperatures used in initial Gibbs states
max_level = 40              # Highest oscillator level
single_basis = 0:max_level

nPts = 20

# Ranges of maximum compression and evolution times
ωmin = 1
ωmax = 3

τα_min = 0.5
τα_max = 10

ωs = range(ωmin, ωmax, length = nPts)
ταs = range(τα_min, τα_max, length = nPts)

# Gas Hamiltonian
H0_gas = diagm(single_basis .+ 1 / 2)
H_scatter_gas = diagm(1 => sqrt.(1:max_level))^2 + diagm(-1 => sqrt.(1:max_level))^2

function H_gas(α)
    return (H0_gas * (1 + α / 2) + α / 4 * H_scatter_gas)
end

for T in Ts
    if !isfile("data/3D_Osc_Engine/compression_expansion/Compression_Expansion_T$(T).jld2")
        res_expansion = zeros(nPts, nPts)
        res_compression = zeros(nPts, nPts)

        Gibbs_expanded = exp(-H_gas(0) ./ T)
        Gibbs_expanded = Gibbs_expanded ./ tr(Gibbs_expanded)

        @showprogress for ii = 1:nPts
            ω = ωs[ii]          # Compressed frequency
            α_max = ω^2 - 1     # Compressed α
            ϵ = 1 / max_level / (2 * π) / (1 + α_max / 2)
            δτ = ϵ / 5

            Gibbs_compressed = exp(-H_gas(α_max) ./ T)
            Gibbs_compressed = Gibbs_compressed ./ tr(Gibbs_compressed)

            adiabatic_compressed_energy = tr(Gibbs_expanded * H_gas(0)) * ω
            adiabatic_expanded_energy = tr(Gibbs_compressed * H_gas(α_max)) / ω

            Threads.@threads for jj = 1:nPts
                τα = ταs[jj]
                nSteps = floor(Int, τα / δτ)
                U = diagm(Vector{ComplexF64}(ones(max_level + 1)))
                # Compression α(t)
                function α(t)
                    return (t / τα * α_max)
                end
                # Compression U
                for kk = 0:(nSteps-1)
                    U = RKstep(t -> H_gas(α(t)), U, kk * δτ, δτ)
                end
                U = RKstep(t -> H_gas(α(t)), U, nSteps * δτ, τα - nSteps * δτ)

                compressed_state = U * Gibbs_expanded * U'
                expanded_state = U' * Gibbs_compressed * U

                compressed_energy = tr(H_gas(α_max) * compressed_state) |> real
                expanded_energy = tr(H_gas(0) * expanded_state) |> real

                res_compression[ii, jj] = compressed_energy / adiabatic_compressed_energy
                res_expansion[ii, jj] = expanded_energy / adiabatic_expanded_energy

                GC.safepoint()
            end
            save_object(
                "data/3D_Osc_Engine/compression_expansion/Compression_Expansion_T$(T).jld2",
                (res_compression, res_expansion, ωs, ταs),
            )
        end
    end
end

## FIGURES
set_theme!(CF_theme)

data = readdir("data/3D_Osc_Engine/compression_expansion", join = true)

fig = Figure(resolution = (1200, 1200))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Adiabaticity measurement",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

main_grid = fig[2:10, 1] = GridLayout()
panels_grid = main_grid[1, 1:9] = GridLayout()

T_grid = [panels_grid[1, ii] for ii in eachindex(data)]

labs = ["(a)", "(b)", "(c)", "(d)"]
Ts_lab = [L"\omega_T = 1", L"\omega_T = 5"]

for ii in eachindex(Ts_lab)
    r = load_object(data[ii])
    ax_compression = Axis(T_grid[ii][1, 1], xticks = 1:3)
    ax_expansion = Axis(T_grid[ii][2, 1], xticks = 1:3)
    if ii == 2
        ax_compression.yaxisposition = :right
        ax_expansion.yaxisposition = :right
    end
    heatmap!(ax_compression, r[3], r[4], r[1], colormap = CF_heat, colorrange = (1, 1.15))
    heatmap!(ax_expansion, r[3], r[4], r[2], colormap = CF_heat, colorrange = (1, 1.15))

    ax_compression.title = Ts_lab[ii]


    ax_expansion.xlabel = L"\omega"
    if ii == 1
        ax_expansion.ylabel = L"\tau_\alpha"
        ax_compression.ylabel = L"\tau_\alpha"
        hidexdecorations!(ax_compression, label = false)
        text!(
            ax_compression,
            0.05,
            0.95,
            text = "(a)",
            align = (:left, :top),
            space = :relative,
            fontsize = 36,
            font = :latex,
            color = :white,
        )
        text!(
            ax_expansion,
            0.05,
            0.95,
            text = "(b)",
            align = (:left, :top),
            space = :relative,
            fontsize = 36,
            font = :latex,
            color = :white,
        )
    end
    if ii == 2
        ax_expansion.ylabel = "Expansion"
        ax_compression.ylabel = "Compression"
        hideydecorations!(ax_compression, label = false)
        hideydecorations!(ax_expansion, label = false)
        hidexdecorations!(ax_compression, label = false)
        text!(
            ax_compression,
            0.05,
            0.95,
            text = "(c)",
            align = (:left, :top),
            space = :relative,
            fontsize = 36,
            font = :latex,
            color = :white,
        )
        text!(
            ax_expansion,
            0.05,
            0.95,
            text = "(d)",
            align = (:left, :top),
            space = :relative,
            fontsize = 36,
            font = :latex,
            color = :white,
        )
    end

end

cb = Colorbar(
    main_grid[1, 10],
    limits = (1, 1.15),
    colormap = CF_heat,
    labelfont = :latex,
    ticklabelfont = :latex,
    ticks = [1, 1.05, 1.1, 1.15],
)

fig

save("Adiabaticity.pdf", fig)
