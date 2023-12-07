include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

# COMBINATION OF (τα,τ,ω)
params = [(1, 1, 2), (4, 10, 2), (1, 1, 3), (4, 10, 3)]

## Parameters
Tc = 1 / 10                 # Cold temperature
Th = 5                      # Hot temperature
max_level = 40              # Max Fock level   
single_basis = 0:max_level  # Single oscillator basis

# Single oscillator Hamiltonian
id = diagm(ones(length(single_basis)))
H0 = diagm(single_basis .+ 1 / 2)
H_scatter = diagm(1 => sqrt.(1:max_level))^2 + diagm(-1 => sqrt.(1:max_level))^2

function H(α)
    return (H0 * (1 + α / 2) + α / 4 * H_scatter)
end

for p in params
    τα = p[1]
    τ = p[2]
    ω = p[3]
    if !isfile("data/3D_Osc_Engine/engine_operation/Engine_τα$(τα)_τ$(τ)_ω$(ω).jld2")

        τ_contact_hot = τ           # Time for contact with the hot mode
        τ_contact_cold = τ          # Time for contact with the cold mode

        nCycles = 50
        # Interaction
        Φ0 = 1      # Interaction strength
        d = 1       # Potential offset
        σ = 1       # Potential width

        α_max = ω^2 - 1     # Maximum α
        # Largest allowed time step for the RK solution
        ϵ = 1 / max_level / (2 * π) / (1 + α_max / 2)
        # Time step for RK solution
        δτ = ϵ / 5

        # Mode coupling
        Φ_g = [mode_interaction(n, m, 1, d, σ) for n in single_basis, m in single_basis]
        Φ_mat = Matrix{Float64}(kron(Φ_g, Φ_g))

        # Bath states
        hot_bath = exp(-H(α_max) ./ Th)
        hot_bath = hot_bath ./ tr(hot_bath)

        cold_bath = exp(-H(0) ./ Tc)
        cold_bath = cold_bath ./ tr(cold_bath)

        ## COMPRESSION U
        nSteps = floor(Int, τα / δτ)
        U_compression = diagm(Vector{ComplexF64}(ones(max_level + 1)))

        # Compression α(t)
        function α(t)
            return (t / τα * α_max)
        end
        # Compression U
        @showprogress for kk = 0:(nSteps-1)
            U_compression = RKstep(t -> H(α(t)), U_compression, kk * δτ, δτ)
        end
        U_compression = RKstep(t -> H(α(t)), U_compression, nSteps * δτ, τα - nSteps * δτ)

        # Hamiltonians for composite systems in compressed and expanded configurations
        H_total_expanded = kron(H(0), id) + kron(id, H(0)) + Φ0 .* Φ_mat
        H_total_compressed = kron(H(α_max), id) + kron(id, H(α_max)) + Φ0 .* Φ_mat

        # Evolution operators for contact with hot and cold baths

        Hot = unitary_evolution(H_total_compressed, τ_contact_hot) |> sparse
        Cold = unitary_evolution(H_total_expanded, τ_contact_cold) |> sparse

        # A function takes a gas state and performs an engine cycle.
        # Post-compression, 
        function cycle(ρ0)
            # Compress the gas
            compressed_cold = U_compression * ρ0 * U_compression'
            # Bring it into contact with the hot bath
            hot_contact = Hot * kron(compressed_cold, hot_bath) * Hot'
            # Trace out the bath
            compressed_hot = partial_trace(hot_contact, 2)
            # Expand the gas
            expanded_hot = U_compression' * compressed_hot * U_compression
            # Bring it into contact with the cold bath
            cold_contact = Cold * kron(expanded_hot, cold_bath) * Cold'
            # Trace out the bath
            expanded_cold = partial_trace(cold_contact, 2)
            return (ρ0, compressed_cold, compressed_hot, expanded_hot, expanded_cold)
        end

        # Calculation
        ρ0 = copy(cold_bath)
        res = Array{NTuple{5,Matrix{ComplexF64}}}(undef, nCycles)
        res[1] = (ρ0, ρ0, ρ0, ρ0, ρ0)

        @showprogress for c = 2:nCycles
            res[c] = cycle(res[c-1][end])
        end

        save_object(
            "data/3D_Osc_Engine/engine_operation/Engine_τα$(τα)_τ$(τ)_ω$(ω).jld2",
            (res, ω, τα, τ),
        )

    end

end

## FIGURES
set_theme!(CF_theme)
data = [
    "data/3D_Osc_Engine/engine_operation/Engine_τα1_τ1_ω3.jld2",
    "data/3D_Osc_Engine/engine_operation/Engine_τα1_τ1_ω2.jld2",
    "data/3D_Osc_Engine/engine_operation/Engine_τα4_τ10_ω3.jld2",
    "data/3D_Osc_Engine/engine_operation/Engine_τα4_τ10_ω2.jld2",
]
fig = Figure(resolution = (1200, 1800))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Engine cycle",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

main_grid = fig[2:20, 1] = GridLayout()

stab_grid = main_grid[1:10, 1] = GridLayout()
stab_legend = main_grid[11, 1] = GridLayout()
energy_grid = main_grid[12:15, 1] = GridLayout()
energy_legend = main_grid[16, 1] = GridLayout()

## Stabilization legend
expanded_legend_marker =
    MarkerElement(color = :black, marker = :circle, markersize = 24, strokecolor = :black)
compressed_legend_marker =
    MarkerElement(color = :black, marker = '◯', markersize = 24, strokecolor = :black)

hot_legend = PolyElement(color = CF_vermillion, strokecolor = :transparent)
cold_legend = PolyElement(color = CF_sky, strokecolor = :transparent)

Legend(
    stab_legend[1, 1],
    [expanded_legend_marker, compressed_legend_marker, hot_legend, cold_legend],
    ["Expanded", "Compressed", "Hot", "Cold"],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    orientation = :horizontal,
    titlevisible = false,
)

## Energy legend
expanded_legend_marker =
    MarkerElement(color = :black, marker = :cross, markersize = 24, strokecolor = :black)
compressed_legend_marker =
    MarkerElement(color = :black, marker = '□', markersize = 24, strokecolor = :black)

ω2_legend = PolyElement(color = CF_orange, strokecolor = :transparent)
ω3_legend = PolyElement(color = CF_green, strokecolor = :transparent)

Legend(
    energy_legend[1, 1],
    [expanded_legend_marker, compressed_legend_marker, ω2_legend, ω3_legend],
    ["Slow", "Fast", L"\omega = 2", L"\omega = 3"],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    orientation = :horizontal,
    titlevisible = false,
)

energy_colors = [CF_green, CF_orange, CF_green, CF_orange]
energy_markers = ['□', '□', :cross, :cross]

ax_fast_large = Axis(
    stab_grid[1, 1],
    title = L"\omega = 3",
    ylabel = "Energy",
    xticks = 0:10:50,
    yticks = 0:3,
)
ax_fast_small = Axis(
    stab_grid[1, 2],
    title = L"\omega = 2",
    ylabel = "Fast",
    xticks = 0:10:50,
    yticks = 0:3,
    yaxisposition = :right,
)
ax_slow_large = Axis(
    stab_grid[2, 1],
    xlabel = "Cycle",
    ylabel = "Energy",
    xticks = 0:10:50,
    yticks = 0:5,
)
ax_slow_small = Axis(
    stab_grid[2, 2],
    ylabel = "Slow",
    xlabel = "Cycle",
    xticks = 0:10:50,
    yticks = 0:5,
    yaxisposition = :right,
)
ax = [ax_fast_large, ax_fast_small, ax_slow_large, ax_slow_small]

ax_efficiency =
    Axis(energy_grid[1, 1], xlabel = "Cycle", ylabel = "Efficiency", xticks = 0:10:50)
ax_power = Axis(energy_grid[1, 2], xlabel = "Cycle", ylabel = "Power", xticks = 0:10:50)
ylims!(ax_power, (0, 0.06))
labs = ["(a)", "(b)", "(c)", "(d)"]

ylims!(ax_fast_large, (0, 3.03))
ylims!(ax_fast_small, (0, 3.03))
ylims!(ax_slow_large, (0, 5.05))
ylims!(ax_slow_small, (0, 5.05))

for ii in eachindex(data)
    d = load_object(data[ii])
    res = d[1]
    ω = d[2]
    α_max = ω^2 - 1
    EC = real.([tr(H(0) * r[1]) for r in res[2:end]])
    CC = real.([tr(H(α_max) * r[2]) for r in res[2:end]])
    CH = real.([tr(H(α_max) * r[3]) for r in res[2:end]])
    EH = real.([tr(H(0) * r[4]) for r in res[2:end]])
    EC_end = real.([tr(H(0) * r[5]) for r in res[2:end]])

    W_in = CC - EC
    Q_in = CH - CC
    W_out = EH - CH
    Q_out = EC_end - EH

    period = 2 * (d[3] + d[4])
    useful_work = -(W_out + W_in)

    scatter!(ax[ii], EC, color = CF_sky, markersize = 12)
    scatter!(ax[ii], CC, color = CF_sky, marker = '◯', markersize = 12)
    scatter!(ax[ii], CH, color = CF_vermillion, marker = '◯', markersize = 12)
    scatter!(ax[ii], EH, color = CF_vermillion, markersize = 12)
    xlims!(ax[ii], (0, 51))

    scatter!(
        ax_efficiency,
        useful_work ./ Q_in,
        color = energy_colors[ii],
        marker = energy_markers[ii],
        markersize = 12,
    )
    scatter!(
        ax_power,
        useful_work ./ period,
        color = energy_colors[ii],
        marker = energy_markers[ii],
        markersize = 12,
    )
    text!(
        ax[ii],
        0.05,
        0.95,
        text = labs[ii],
        align = (:left, :top),
        space = :relative,
        fontsize = 36,
        font = :latex,
        color = :black,
    )
end

hlines!(ax_efficiency, [1 / 2, 2 / 3], color = CF_red, linewidth = 4)
hidexdecorations!(ax_fast_large, label = false)
hidexdecorations!(ax_fast_small, label = false)

hideydecorations!(ax_slow_small, label = false)
hideydecorations!(ax_fast_small, label = false)

text!(
    ax_efficiency,
    0.85,
    0.8,
    text = "(e)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)

text!(
    ax_power,
    0.85,
    0.8,
    text = "(f)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)

fig
save("Engine_Operation.pdf", fig)
