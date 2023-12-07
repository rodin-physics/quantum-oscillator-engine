include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

## Parameters
function thermal_contact(Tg, Tb, nPts, ω, δτ, σ, d, max_level, Φs)

    single_basis = 0:max_level
    α = ω^2 - 1

    # Gas Hamiltonian
    H0_gas = diagm(single_basis .+ 1 / 2)
    H_scatter_gas = diagm(1 => sqrt.(1:max_level))^2 + diagm(-1 => sqrt.(1:max_level))^2
    H_osc = H0_gas * (1 + α / 2) + α / 4 * H_scatter_gas
    id = diagm(ones(length(single_basis)))

    # States
    Gibbs_gas = exp(-H_osc ./ Tg)
    Gibbs_gas = Gibbs_gas ./ tr(Gibbs_gas)

    Gibbs_bath = exp(-H_osc ./ Tb)
    Gibbs_bath = Gibbs_bath ./ tr(Gibbs_bath)

    # Mode coupling
    Φ_g = [mode_interaction(n, m, 1, d, σ) for n in single_basis, m in single_basis]
    Φ_mat = Matrix{Float64}(kron(Φ_g, Φ_g))

    tr_dist = zeros(nPts)
    ρ_tot = kron(Gibbs_gas, Gibbs_bath)

    Φ0 = 1
    H_total = kron(H_osc, id) + kron(id, H_osc) + Φ0 .* Φ_mat
    U = unitary_evolution(H_total, δτ) |> sparse
    tr_dist[1] = real(trace_distance(Gibbs_gas, Gibbs_bath))

    @showprogress for ii = 2:nPts
        if mod(ii, 10) == 0
            ρ_tot = kron(partial_trace(ρ_tot, 2), Gibbs_bath)
        end

        if haskey(Φs, ii)
            Φ0 = get(Φs, ii, nothing)
            H_total = kron(H_osc, id) + kron(id, H_osc) + Φ0 .* Φ_mat
            U = unitary_evolution(H_total, δτ) |> sparse
        end

        ρ_tot = U * ρ_tot * U'
        ρ_gas = partial_trace(ρ_tot, 2)

        tr_dist[ii] = real(trace_distance(ρ_gas, Gibbs_bath))

    end
    ρ_gas = partial_trace(ρ_tot, 2)

    return (tr_dist, ρ_gas, Tg, Tb, ω, δτ, σ, d)
end

# Parameters
d = 1       # Potential offset
σ = 1       # Potential width

max_level = 40
ω = 3       # Compressed frequency

Th = 5      # Hot temperature
Tc = 1      # Cold temperature
nPts = 120  # Number of steps

δτ = 5      # Step size

# Adjusting the interaction strength
Φs = Dict(50 => 1 / 5, 90 => 1 / 20)

params = [(Th, Tc, 1), (Tc, Th, 1), (Th, Tc, ω), (Tc, Th, ω)]
Threads.@threads for p in params
    Tg = p[1]
    Tb = p[2]
    ω = p[3]
    if !isfile(
        "data/3D_Osc_Engine/thermal_contact/thermal_contact_Tg$(Tg)_Tb$(Tb)_ω$(ω)_σ$(σ).jld2",
    )
        r = thermal_contact(Tg, Tb, nPts, ω, δτ, σ, d, max_level, Φs)
        save_object(
            "data/3D_Osc_Engine/thermal_contact/thermal_contact_Tg$(Tg)_Tb$(Tb)_ω$(ω)_σ$(σ).jld2",
            r,
        )
    end
end

## FIGURE
set_theme!(CF_theme)
heating_expanded =
    load_object("data/3D_Osc_Engine/thermal_contact/thermal_contact_Tg1_Tb5_ω1_σ1.jld2")
cooling_expanded =
    load_object("data/3D_Osc_Engine/thermal_contact/thermal_contact_Tg5_Tb1_ω1_σ1.jld2")
heating_compressed =
    load_object("data/3D_Osc_Engine/thermal_contact/thermal_contact_Tg1_Tb5_ω3_σ1.jld2")
cooling_compressed =
    load_object("data/3D_Osc_Engine/thermal_contact/thermal_contact_Tg5_Tb1_ω3_σ1.jld2")
fig = Figure(resolution = (1200, 600))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Thermal equilibration",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

main_grid = fig[2:10, 1] = GridLayout()
data_grid = main_grid[1:9, 1] = GridLayout()
legend_grid = main_grid[10, 1] = GridLayout()

ax_eq = Axis(
    data_grid[1, 1],
    xlabel = L"\tau",
    ylabel = L"T\,\left[\rho_g(\tau), \rho_b\right]",
    yscale = log10,
)
ax_dist = Axis(
    data_grid[1, 2],
    xlabel = L"\omega_T",
    ylabel = L"T\,\left[\rho_g, \rho(\omega_T)\right]",
)
ax_eq.title = "Equilibration process"
ax_dist.title = "Final state"

labs = ["(a)", "(b)"]

scatter!(
    ax_eq,
    δτ .* (eachindex(heating_expanded[1]) .- 1),
    heating_expanded[1],
    color = CF_vermillion,
    markersize = 12,
    # linewidth = 4,
)
scatter!(
    ax_eq,
    δτ .* (eachindex(cooling_expanded[1]) .- 1),
    cooling_expanded[1],
    color = CF_sky,
    markersize = 12,
    # linewidth = 4,
)
scatter!(
    ax_eq,
    δτ .* (eachindex(heating_compressed[1]) .- 1),
    heating_compressed[1],
    color = CF_vermillion,
    marker = '◯',
    markersize = 12,
    # linestyle = :dot,
    # linewidth = 4,
)
scatter!(
    ax_eq,
    δτ .* (eachindex(cooling_compressed[1]) .- 1),
    cooling_compressed[1],
    color = CF_sky,
    marker = '◯',
    markersize = 12,
    # linestyle = :dot,
    # linewidth = 4,
)
vlines!(
    ax_eq,
    δτ .* [10, 20, 30, 40, 60, 70, 80, 100, 110],
    color = CF_black,
    linestyle = :dash,
)
vlines!(ax_eq, δτ .* [50, 90], color = CF_black)

Ts = range(0.1, 10, length = 1000)

ds_heating_expanded = zeros(length(Ts))
ds_cooling_expanded = zeros(length(Ts))
ds_heating_compressed = zeros(length(Ts))
ds_cooling_compressed = zeros(length(Ts))

# Gas Hamiltonian
H0_gas = diagm(single_basis .+ 1 / 2)
H_scatter_gas = diagm(1 => sqrt.(1:max_level))^2 + diagm(-1 => sqrt.(1:max_level))^2
α_max = ω^2 - 1
H_osc = H0_gas * (1 + α_max / 2) + α_max / 4 * H_scatter_gas

for ii in eachindex(Ts)
    gas_compressed = exp(-H_osc ./ Ts[ii])
    gas_compressed = gas_compressed ./ tr(gas_compressed)
    gas_expanded = exp(-H0_gas ./ Ts[ii])
    gas_expanded = gas_expanded ./ tr(gas_expanded)

    ds_heating_expanded[ii] = real(trace_distance(heating_expanded[2], gas_expanded))
    ds_cooling_expanded[ii] = real(trace_distance(cooling_expanded[2], gas_expanded))
    ds_heating_compressed[ii] = real(trace_distance(heating_compressed[2], gas_compressed))
    ds_cooling_compressed[ii] = real(trace_distance(cooling_compressed[2], gas_compressed))

end

lines!(ax_dist, Ts, ds_heating_expanded, color = CF_vermillion, linewidth = 4)
lines!(
    ax_dist,
    Ts,
    ds_heating_compressed,
    color = CF_vermillion,
    linewidth = 4,
    linestyle = :dot,
)
lines!(ax_dist, Ts, ds_cooling_expanded, color = CF_sky, linewidth = 4)
lines!(ax_dist, Ts, ds_cooling_compressed, color = CF_sky, linewidth = 4, linestyle = :dot)

vlines!(ax_dist, [Tc], color = CF_sky)
vlines!(ax_dist, [Th], color = CF_vermillion)

xlims!(ax_eq, (0, 600))
ylims!(ax_eq, (1e-3, 1))

xlims!(ax_dist, (0, 10.25))
ylims!(ax_dist, (0, 0.8))

text!(
    ax_eq,
    0.9,
    0.98,
    text = "(a)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)

text!(
    ax_dist,
    0.9,
    0.98,
    text = "(b)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)

expanded_legend_marker =
    MarkerElement(color = :black, marker = :circle, markersize = 24, strokecolor = :black)
compressed_legend_marker =
    MarkerElement(color = :black, marker = '◯', markersize = 24, strokecolor = :black)

heat_legend = PolyElement(color = CF_vermillion, strokecolor = :transparent)
cool_legend = PolyElement(color = CF_sky, strokecolor = :transparent)

expanded_legend = LineElement(color = :black, linewidth = 4, linestyle = :solid)
compressed_legend = LineElement(color = :black, linewidth = 4, linestyle = :dot)


Legend(
    legend_grid[1, 1],
    [
        expanded_legend_marker,
        compressed_legend_marker,
        heat_legend,
        cool_legend,
        expanded_legend,
        compressed_legend,
    ],
    ["Expanded", "Compressed", "Heat", "Cool", "Expanded", "Compressed"],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    titlefont = :boldlatex,
    orientation = :horizontal,
    titlevisible = false,
)
fig
save("Thermal_contact.pdf", fig)
