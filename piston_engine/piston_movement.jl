include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

# Interaction properties
d = 0       # Potential offset
l = 1       # Hamonic oscillator length
Φ0 = -5     # Potential magnitude

σs = [1 / 4, 1 / 2, 1, 2]  # Potential widths

# Oscillator properties
max_level = 60
single_basis = 0:max_level
ϵ = 1 / max_level / (2 * π)
δτ = ϵ / 5

# Piston properties
nτs = 50
τ_min = 1 / 10
τ_max = 10
τs = exp.(range(log(τ_min), log(τ_max), length = nτs))

for σ in σs
    if !isfile("data/piston_engine/piston_movement/Piston_Movement_$(σ).jld2")
        y_min = 0
        y_max = 10 * σ
        # Hamiltonian
        H_osc = diagm(single_basis .+ 1 / 2)
        Φ_mat = Matrix{Float64}([
            mode_interaction(n, m, l, d, σ) for n in single_basis, m in single_basis
        ])

        retracted_states =
            eigen(Matrix{Float64}(H_osc + Φ0 * Φ_mat * exp(-y_max^2 / 2 / σ^2)))
        advanced_states =
            eigen(Matrix{Float64}(H_osc + Φ0 * Φ_mat * exp(-y_min^2 / 2 / σ^2)))

        ground_state_retracted = retracted_states.vectors[:, 1]
        ground_energy_retracted = retracted_states.values[1]

        ground_state_advanced = advanced_states.vectors[:, 1]
        ground_energy_advanced = advanced_states.values[1]

        res_retraction = zeros(nτs)
        res_advance = zeros(nτs)
        pr = Progress(nτs)

        Threads.@threads for ii = 1:nτs
            τ_piston = τs[ii]
            nSteps = floor(Int, τ_piston / δτ)

            state_retraction = copy(ground_state_advanced)
            state_advance = copy(ground_state_retracted)

            function piston(τ, y_init, y_final)
                res = y_init + τ * (y_final - y_init) / τ_piston
                return res
            end

            function H_retraction(t)
                res = H_osc + Φ0 * Φ_mat * exp(-piston(t, y_min, y_max)^2 / 2 / σ^2)
                return res
            end

            function H_advance(t)
                res = H_osc + Φ0 * Φ_mat * exp(-piston(t, y_max, y_min)^2 / 2 / σ^2)
                return res
            end

            for kk = 0:(nSteps-1)
                state_retraction =
                    RKstep(t -> H_retraction(t), state_retraction, kk * δτ, δτ)
                state_advance = RKstep(t -> H_advance(t), state_advance, kk * δτ, δτ)
            end
            state_retraction = RKstep(
                t -> H_retraction(t),
                state_retraction,
                nSteps * δτ,
                τ_piston - nSteps * δτ,
            )
            state_advance = RKstep(
                t -> H_advance(t),
                state_advance,
                nSteps * δτ,
                τ_piston - nSteps * δτ,
            )

            res_retraction[ii] =
                real.(state_retraction' * H_retraction(τ_piston) * state_retraction)
            res_advance[ii] = real.(state_advance' * H_advance(τ_piston) * state_advance)
            next!(pr)
            GC.safepoint()
        end
        save_object(
            "data/piston_engine/piston_movement/Piston_Movement_$(σ).jld2",
            (
                τs,
                res_retraction,
                res_advance,
                σ,
                ground_energy_retracted,
                ground_energy_advanced,
                ground_state_advanced' *
                (H_osc + Φ0 * Φ_mat * exp(-y_max^2 / 2 / σ^2)) *
                ground_state_advanced,
                ground_state_retracted' *
                (H_osc + Φ0 * Φ_mat * exp(-y_min^2 / 2 / σ^2)) *
                ground_state_retracted,
            ),
        )
    end
end

## FIGURES
set_theme!(CF_theme)
colors = [CF_vermillion, CF_green, CF_sky, CF_blue]

# data = readdir("data/piston_engine/piston_movement", join = true)
data = [
    "data/piston_engine/piston_movement/Piston_Movement_0.25.jld2",
    "data/piston_engine/piston_movement/Piston_Movement_0.5.jld2",
    "data/piston_engine/piston_movement/Piston_Movement_1.0.jld2",
    "data/piston_engine/piston_movement/Piston_Movement_2.0.jld2",
]
fig = Figure(size = (1200, 600))

supertitle = fig[1, 1:2]
Label(
    supertitle,
    "Adiabaticity measurement",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

ax_retraction = Axis(
    fig[2:10, 1],
    xscale = log10,
    xlabel = L"\tau_p",
    ylabel = "Energy",
    title = "Retraction",
    xticks = [0.1, 1, 10],
    yticks = 0.5:0.2:1.1,
)
ax_advance = Axis(
    fig[2:10, 2],
    xscale = log10,
    xlabel = L"\tau_p",
    ylabel = "Energy",
    title = "Advance",
    xticks = [0.1, 1, 10],
    yticks = -4:1:-1,
)

legend = fig[11, 1:2] = GridLayout()

for idx in eachindex(data)
    dt = load_object(data[idx])
    scatter!(ax_retraction, dt[1], dt[2], color = colors[idx])
    scatter!(ax_advance, dt[1], dt[3], color = colors[idx])
    hlines!(ax_retraction, [dt[5]], color = colors[idx], linewidth = 2)
    hlines!(ax_advance, [dt[6]], color = colors[idx], linewidth = 2)
    hlines!(ax_retraction, [dt[7]], color = colors[idx], linewidth = 2, linestyle = :dash)
    hlines!(ax_advance, [dt[8]], color = colors[idx], linewidth = 2, linestyle = :dash)

end

text!(
    ax_retraction,
    0.85,
    0.95,
    text = "(a)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)

text!(
    ax_advance,
    0.85,
    0.95,
    text = "(b)",
    align = (:left, :top),
    space = :relative,
    fontsize = 36,
    font = :latex,
    color = :black,
)


σ_markers = [PolyElement(color = c, strokecolor = :transparent) for c in colors]
σ_labs = ["1/4", "1/2", "1", "2"]

adiabatic_marker = LineElement(color = :black)
sudden_marker = LineElement(color = :black, linestyle = :dash)

Legend(
    legend[1, 1],
    [[adiabatic_marker, sudden_marker], σ_markers],
    [["Adiabatic", "Instantaneous"], σ_labs],
    ["", L"\sigma:"],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    orientation = :horizontal,
    titlevisible = false,
    titleposition = :left,
)

fig
save("Piston_movement.pdf", fig)
