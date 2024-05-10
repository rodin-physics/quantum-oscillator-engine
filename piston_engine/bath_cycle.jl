include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

# COMBINATION OF (τ_stroke, σ)
params = [(5, 1 / 2), (5, 2), (10, 1 / 2), (10, 2)]

## Parameters
Tc = 1 / 10                 # Cold temperature
Th = 5                      # Hot temperature
max_level = 50              # Max Fock level   
single_basis = 0:max_level  # Single oscillator basis
ϵ = 1 / max_level / (2 * π)
δτ = ϵ / 5
nCycles = 80

Φ0 = -5
# SINGLE OSCILLATOR
id = diagm(ones(length(single_basis)))
H0 = diagm(single_basis .+ 1 / 2)

# BATH COUPLING
Y0 = 1      # Interaction strength
d = 1       # Potential offset
λ = 1       # Potential width

Y_g = [mode_interaction(n, m, 1, d, λ) for n in single_basis, m in single_basis]
Y_mat = Matrix{Float64}(kron(Y_g, Y_g))
H_bath = kron(H0, id) + kron(id, H0) + Y0 .* Y_mat

# BATH STATES
hot_bath = exp(-H0 ./ Th)
hot_bath = hot_bath ./ tr(hot_bath)

cold_bath = exp(-H0 ./ Tc)
cold_bath = cold_bath ./ tr(cold_bath)

for p in params
    τ_stroke = p[1]
    σ = p[2]        # Piston-oscillator interaction width
    Bath_U = unitary_evolution(H_bath, τ_stroke) |> sparse

    # PISTON COUPLING
    println("Calculating Φ")
    Φ_mat = Matrix{Float64}([
        mode_interaction(n, m, 1, 0, σ) for n in single_basis, m in single_basis
    ])
    nSteps = floor(Int, τ_stroke / δτ)

    y_min = 0
    y_max = 5 * σ

    function piston(τ, y_init, y_final)
        res = y_init + τ * (y_final - y_init) / τ_stroke
        return res
    end

    function H_retraction(t)
        res = H0 + Φ0 * Φ_mat * exp(-piston(t, y_min, y_max)^2 / 2 / σ^2)
        return res
    end

    println("Calculating U...")
    if !isfile("data/piston_engine/piston_movement/U_retraction_τ$(τ_stroke)_σ$(σ).jld2")
        U_retraction = diagm(ones(length(single_basis)))
        @showprogress for kk = 0:(nSteps-1)
            U_retraction = RKstep(t -> H_retraction(t), U_retraction, kk * δτ, δτ)
        end

        U_retraction =
            RKstep(t -> H_retraction(t), U_retraction, nSteps * δτ, τ_stroke - nSteps * δτ)
        save_object(
            "data/piston_engine/piston_movement/U_retraction_τ$(τ_stroke)_σ$(σ).jld2",
            U_retraction,
        )

    else
        U_retraction = load_object(
            "data/piston_engine/piston_movement/U_retraction_τ$(τ_stroke)_σ$(σ).jld2",
        )

    end
    U_retraction = Matrix{ComplexF64}(U_retraction)
    println("Starting Calculations")
    if !isfile("data/piston_engine/bath_cycle/Bath_cycle_τ$(τ_stroke)_σ$(σ).jld2")
        function cycle(ρ0)
            # Heat the gas
            hot_contact = Bath_U * sparse(kron(ρ0, hot_bath)) * Bath_U'
            # Trace out the bath
            close_hot = partial_trace(hot_contact, 2)
            # Retract the piston
            far_hot = U_retraction * close_hot * U_retraction'
            # Bring it into contact with the cold bath
            cold_contact = Bath_U * sparse(kron(far_hot, cold_bath)) * Bath_U'
            # Trace out the bath
            far_cold = partial_trace(cold_contact, 2)
            # Advance the piston
            close_cold = U_retraction' * far_cold * U_retraction
            return (ρ0, close_hot, far_hot, far_cold, close_cold)
        end

        # Calculation
        # Cold, piston close
        ρ0 = exp(-(H0 + Φ0 * Φ_mat) ./ Tc)
        ρ0 = ρ0 ./ tr(ρ0)
        # ρ0 = copy(cold_bath)
        res = Array{NTuple{5,Matrix{ComplexF64}}}(undef, nCycles)
        res[1] = (ρ0, ρ0, ρ0, ρ0, ρ0)

        @showprogress for c = 2:nCycles
            res[c] = cycle(res[c-1][end])
        end

        save_object(
            "data/piston_engine/bath_cycle/Bath_cycle_τ$(τ_stroke)_σ$(σ).jld2",
            (res, τ_stroke, σ),
        )

    end

end

## FIGURES
set_theme!(CF_theme)
data = [
    "data/piston_engine/bath_cycle/Bath_cycle_τ5_σ0.5.jld2",
    "data/piston_engine/bath_cycle/Bath_cycle_τ5_σ2.jld2",
    "data/piston_engine/bath_cycle/Bath_cycle_τ10_σ0.5.jld2",
    "data/piston_engine/bath_cycle/Bath_cycle_τ10_σ2.jld2",
]

fig = Figure(size = (1200, 1800))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Bath-powered cycle",
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
retracted_legend_marker =
    MarkerElement(color = :black, marker = :circle, markersize = 24, strokecolor = :black)
advanced_legend_marker =
    MarkerElement(color = :black, marker = '◯', markersize = 24, strokecolor = :black)

hot_legend = PolyElement(color = CF_vermillion, strokecolor = :transparent)
cold_legend = PolyElement(color = CF_sky, strokecolor = :transparent)

Legend(
    stab_legend[1, 1],
    [retracted_legend_marker, advanced_legend_marker, hot_legend, cold_legend],
    ["Retracted", "Advanced", "Hot", "Cold"],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    orientation = :horizontal,
    titlevisible = false,
)

## Energy legend
slow_legend_marker =
    MarkerElement(color = :black, marker = :cross, markersize = 24, strokecolor = :black)
fast_legend_marker =
    MarkerElement(color = :black, marker = '□', markersize = 24, strokecolor = :black)

σ1_2_legend = PolyElement(color = CF_green, strokecolor = :transparent)
σ2_legend = PolyElement(color = CF_orange, strokecolor = :transparent)

Legend(
    energy_legend[1, 1],
    [slow_legend_marker, fast_legend_marker, σ1_2_legend, σ2_legend],
    ["Slow", "Fast", L"\sigma = 1 / 2", L"\sigma = 2"],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    orientation = :horizontal,
    titlevisible = false,
)

# Data 
energy_colors = [CF_green, CF_orange, CF_green, CF_orange]
energy_markers = ['□', '□', :cross, :cross]

ax_fast_narrow =
    Axis(stab_grid[1, 1], title = L"\sigma = 1/2", ylabel = "Energy", xticks = 0:10:80)
ax_fast_wide = Axis(
    stab_grid[1, 2],
    title = L"\sigma = 2",
    ylabel = "Fast",
    xticks = 0:10:80,
    yaxisposition = :right,
)
ax_slow_narrow =
    Axis(stab_grid[2, 1], xlabel = "Cycle", ylabel = "Energy", xticks = 0:10:80)
ax_slow_wide = Axis(
    stab_grid[2, 2],
    ylabel = "Slow",
    xlabel = "Cycle",
    xticks = 0:10:80,
    yaxisposition = :right,
)
ax = [ax_fast_narrow, ax_fast_wide, ax_slow_narrow, ax_slow_wide]

ax_efficiency =
    Axis(energy_grid[1, 1], xlabel = "Cycle", ylabel = "Efficiency", xticks = 0:10:80)
ax_power = Axis(energy_grid[1, 2], xlabel = "Cycle", ylabel = "Power", xticks = 0:10:80)

labs = ["(a)", "(b)", "(c)", "(d)"]

ylims!(ax_fast_narrow, (-4.5, 5.05))
ylims!(ax_fast_wide, (-4.5, 5.05))
ylims!(ax_slow_narrow, (-4.5, 5.05))
ylims!(ax_slow_wide, (-4.5, 5.05))
for ii in eachindex(data)
    dt = load_object(data[ii])
    res = dt[1]
    τ_stroke = dt[2]
    σ = dt[3]

    y_min = 0
    y_max = 5 * σ

    Φ_mat = [mode_interaction(n, m, 1, 0, σ) for n in single_basis, m in single_basis]

    # Calculate the system energies at different stages from the numerical results
    H_retracted = Matrix{ComplexF64}(H0 + Φ0 * Φ_mat * exp(-y_max^2 / 2 / σ^2))
    H_advanced = Matrix{ComplexF64}(H0 + Φ0 * Φ_mat * exp(-y_min^2 / 2 / σ^2))

    advanced_cold = real.([tr(H_advanced * r[1]) for r in res[2:end]])
    advanced_hot = real.([tr(H_advanced * r[2]) for r in res[2:end]])
    retracted_hot = real.([tr(H_retracted * r[3]) for r in res[2:end]])
    retracted_cold = real.([tr(H_retracted * r[4]) for r in res[2:end]])
    advanced_cold_end = real.([tr(H_advanced * r[5]) for r in res[2:end]])

    Q_in = advanced_hot - advanced_cold
    W_in = retracted_hot - advanced_hot
    Q_out = retracted_cold - retracted_hot
    W_out = advanced_cold_end - retracted_cold
    period = 4 * τ_stroke
    useful_work = -(W_out + W_in)

    scatter!(ax[ii], advanced_cold, color = CF_sky, marker = '◯', markersize = 12)
    scatter!(ax[ii], advanced_hot, color = CF_vermillion, marker = '◯', markersize = 12)
    scatter!(ax[ii], retracted_hot, color = CF_vermillion, markersize = 12)
    scatter!(ax[ii], retracted_cold, color = CF_sky, markersize = 12)

    xlims!(ax[ii], (-1, 81))

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

    ρH = exp(-H_advanced / Th) / tr(exp(-H_advanced / Th))
    ρC = exp(-H_retracted / Tc) / tr(exp(-H_retracted / Tc))

    η = -(tr((ρH - ρC) * (H_retracted - H_advanced))) / (tr(H_advanced * (ρH - ρC)))

    hlines!(ax_efficiency, [η], color = energy_colors[ii], linewidth = 4)

end

hidexdecorations!(ax_fast_narrow, label = false)
hidexdecorations!(ax_fast_wide, label = false)

hideydecorations!(ax_slow_wide, label = false)
hideydecorations!(ax_fast_wide, label = false)

text!(
    ax_efficiency,
    0.85,
    0.9,
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
    0.9,
    text = "(f)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)

fig
save("Bath_Engine_Operation.pdf", fig)
