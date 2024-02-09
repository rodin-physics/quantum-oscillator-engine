include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

## Parameters
σ = 0.5
δτ = 0.001
nPts = 1000
Φ0 = -10

fock_basis, Φ_mat = load_object("data/2D/Φ_mat_2D_λ1_σ$(σ)_levels50.jld2");

# Optionally reduce the number of Fock states
max_level = 50

idx = findall(x -> x[1] <= max_level && x[2] <= max_level, fock_basis)
fock_basis = fock_basis[idx]
energy_diagonal = [(x + 1 / 2) + (y + 1 / 2) for (x, y) in fock_basis]
Φ_mat = Φ_mat[idx, idx] * Φ0
H = diagm(energy_diagonal) + Φ_mat

U = unitary_evolution(H, δτ) |> sparse

# Initial state
Ψ1_Fock = zeros(max_level + 1)
Ψ1_Fock[1] = 1
Ψ2_Fock = zeros(max_level + 1)
Ψ2_Fock[1] = 1

Ψ0 = kron(Ψ1_Fock, Ψ2_Fock)
Ψ0 = normalize(Ψ0)

current_state = copy(Ψ0)

# Calculate the state evolution
states = zeros(ComplexF64, length(current_state), nPts)
states[:, 1] = current_state
@showprogress for ii = 2:nPts
    states[:, ii] = U * states[:, ii-1]
end

# ENERGY
# System energy as a function of time
E_tot = [states[:, ii]' * H * states[:, ii] for ii = 1:nPts]
# Interaction energy as a function of time
Interaction = [states[:, ii]' * Φ_mat * states[:, ii] for ii = 1:nPts]
# Oscillator energies at all times
Osc_E = [states[:, ii]' * diagm(energy_diagonal) * states[:, ii] for ii = 1:nPts]

# Projection energies
Φ_val = diag(Φ_mat) |> collect |> real  # Interaction energies of two-oscillator eigenstates
H_val = diag(H) |> collect |> real      # Total energies of two-oscillator eigenstates

State_prob = abs.(states) .^ 2          # Probabilities of each Fock state pair

Interaction_Projected = [dot(State_prob[:, ii], Φ_val) for ii = 1:nPts] |> real
E_tot_Projected = [dot(State_prob[:, ii], H_val) for ii = 1:nPts] |> real

Work = Osc_E .- 1 |> real
Heat_input_1 = -Interaction |> real
Heat_input_2 = Interaction_Projected - Interaction |> real

η_single = Work ./ Heat_input_1
η_double = Work ./ Heat_input_2

## FIGURES
set_theme!(CF_theme)
## PROBABILITY DISTRIBUTION
Fock_xs = [v[1] for v in fock_basis]
Fock_ys = [v[2] for v in fock_basis]
xs = range(-5, 5, length = 100) |> collect

fig = Figure(resolution = (1200, 2000))

main_grid = fig[2, 1] = GridLayout()
supertitle = fig[1, 1]
Label(
    supertitle,
    "Perpendicular oscillators",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

rowsize!(fig.layout, 1, Relative(1 / 40))

wf_grid = main_grid[1:2, 1]
energy_grid = main_grid[3, 1]
state_grid = main_grid[4, 1]
efficiency_grid = main_grid[5, 1]

fock_grid = wf_grid[1, 1:19] = GridLayout()
real_grid = wf_grid[2, 1:19] = GridLayout()

fock_colorbar = wf_grid[1, 20] = GridLayout()
real_colorbar = wf_grid[2, 20] = GridLayout()

τs_lab = [L"\tau = 0", L"1/16", L"2/16", L"3/16"]
time_step = [1, floor(Int, 1000 / 16), floor(Int, 1000 / 8), floor(Int, 3 * 1000 / 16)]

ax_fock = [
    Axis(fock_grid[1, ii], xaxisposition = :bottom, aspect = DataAspect()) for
    ii in eachindex(τs_lab)
]
ax_real = [Axis(real_grid[1, ii], aspect = DataAspect()) for ii in eachindex(τs_lab)]

ax_energy = Axis(energy_grid[1, 1])
ax_state = Axis(state_grid[1, 1])

ax_efficiency = Axis(efficiency_grid[1, 1])

lab_fock = ["(a)", "(b)", "(c)", "(d)"]
lab_real = ["(e)", "(f)", "(g)", "(h)"]

## STATES
for ii in eachindex(τs_lab)
    state = (states[:, time_step[ii]])
    fock = abs.(state) .^ 2
    heatmap!(
        ax_fock[ii],
        Fock_xs,
        Fock_ys,
        fock,
        colormap = CF_heat,
        colorrange = (0, 1 / 2),
    )

    rs = abs.(real_space(state, fock_basis, xs, 1)) .^ 2
    heatmap!(ax_real[ii], xs, xs, rs, colormap = CF_heat, colorrange = (0, 1 / 2))

    xlims!(ax_fock[ii], (-0.5, 10.5))
    ylims!(ax_fock[ii], (-0.5, 10.5))
    ax_fock[ii].title = τs_lab[ii]

    text!(
        ax_fock[ii],
        0.75,
        0.95,
        text = lab_fock[ii],
        align = (:left, :top),
        space = :relative,
        fontsize = 36,
        font = :latex,
        color = :white,
    )

    text!(
        ax_real[ii],
        0.75,
        0.95,
        text = lab_real[ii],
        align = (:left, :top),
        space = :relative,
        fontsize = 36,
        font = :latex,
        color = :white,
    )
end
for ii in eachindex(τs_lab)[2:end]
    hideydecorations!(ax_fock[ii], label = false)
    hideydecorations!(ax_real[ii], label = false)
end
ax_fock[1].ylabel = L"k"
ax_real[1].ylabel = L"x_2"

ax_fock[1].xlabel = L"j"
ax_real[1].xlabel = L"x_1"

ax_fock[end].yaxisposition = :right
ax_real[end].yaxisposition = :right

cb_fock = Colorbar(
    wf_grid[1, 20],
    limits = (0, 1 / 2),
    colormap = CF_heat,
    label = "Fock space",
    labelfont = :latex,
    ticklabelfont = :latex,
    ticks = [0, 0.5],
)

cb_real = Colorbar(
    wf_grid[2, 20],
    limits = (0, 1 / 2),
    colormap = CF_heat,
    label = "Real space",
    labelfont = :latex,
    ticklabelfont = :latex,
    ticks = [0, 0.5],
)

## ENERGY
lines!(
    ax_energy,
    δτ .* collect(1:nPts),
    real.(E_tot),
    color = CF_green,
    linewidth = 4,
    label = L"\langle\hat{H}\rangle",
)

lines!(
    ax_energy,
    δτ .* collect(1:nPts),
    real.(Interaction),
    color = CF_vermillion,
    linewidth = 4,
    label = L"\langle\hat{\Phi}\rangle",
)

lines!(
    ax_energy,
    δτ .* collect(1:nPts),
    real.(Osc_E),
    color = CF_sky,
    linewidth = 4,
    label = L"\langle\hat{H}_0\rangle",
)

lines!(
    ax_energy,
    δτ .* collect(1:nPts),
    Interaction_Projected,
    color = CF_vermillion,
    linewidth = 4,
    linestyle = :dash,
    label = L"\bar{\Phi}",
)

lines!(
    ax_energy,
    δτ .* collect(1:nPts),
    real.(Osc_E) + Interaction_Projected,
    color = CF_green,
    linewidth = 4,
    linestyle = :dash,
    label = L"\bar{H}",
)

xlims!(ax_energy, (0, 1.01))
ax_energy.xticks = 0:0.5:1
ax_energy.xlabel = L"\tau"
ax_energy.ylabel = "Energy"
axislegend(ax_energy, orientation = :horizontal, framevisible = false, position = :rb)
ylims!(ax_energy, (-12, 5))

text!(
    ax_energy,
    0.01,
    0.2,
    text = "(i)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    # color = :white,
)


## STATE OCCUPANCY

lines!(
    ax_state,
    δτ .* collect(1:nPts),
    State_prob[findfirst([x == (0, 0) for x in fock_basis]), :],
    color = CF_sky,
    linewidth = 4,
    label = L"|0,0\rangle",
)

lines!(
    ax_state,
    δτ .* collect(1:nPts),
    State_prob[findfirst([x == (2, 0) for x in fock_basis]), :],
    color = CF_vermillion,
    linewidth = 4,
    label = L"|2,0\rangle",
)

lines!(
    ax_state,
    δτ .* collect(1:nPts),
    State_prob[findfirst([x == (1, 1) for x in fock_basis]), :],
    color = CF_green,
    linewidth = 4,
    label = L"|1,1\rangle",
)

lines!(
    ax_state,
    δτ .* collect(1:nPts),
    State_prob[findfirst([x == (2, 2) for x in fock_basis]), :],
    color = CF_red,
    linewidth = 4,
    label = L"|2,2\rangle",
)

xlims!(ax_state, (0, 1.01))
ax_state.xticks = 0:0.5:1
ax_state.xlabel = L"\tau"
ax_state.ylabel = "Probability"
axislegend(ax_state, orientation = :horizontal, framevisible = false, position = :rb)
ylims!(ax_state, (-0.25, 1.01))

text!(
    ax_state,
    0.01,
    0.2,
    text = "(j)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    # color = :white,
)

## EFFICIENCY

lines!(
    ax_efficiency,
    δτ .* collect(1:nPts),
    η_single,
    color = CF_sky,
    linewidth = 4,
    label = L"\eta_\mathrm{single}",
)

lines!(
    ax_efficiency,
    δτ .* collect(1:nPts),
    η_double,
    color = CF_vermillion,
    linewidth = 4,
    label = L"\eta_\mathrm{double}",
)

xlims!(ax_efficiency, (0, 1.01))
ax_state.xticks = 0:0.5:1
ax_efficiency.xlabel = L"\tau"
ax_efficiency.ylabel = L"\eta"
axislegend(ax_efficiency, orientation = :horizontal, framevisible = false, position = :rb)
ylims!(ax_efficiency, (-0.25, 1.01))

text!(
    ax_efficiency,
    0.01,
    0.2,
    text = "(k)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    # color = :white,
)


fig

save("2D_Φ0$(Φ0)_σ$(σ).pdf", fig)

# s1 = sum(H[1,:] .* H[:,1] .* diag(Φ_mat)) - (H*H)[1,1] * Φ_mat[1,1]
# s2 = sum(H[1,:] .* H[:,1] .* energy_diagonal) - (H*H)[1,1] * energy_diagonal[1]

# 1/(1+s1/s2)
# η_double
