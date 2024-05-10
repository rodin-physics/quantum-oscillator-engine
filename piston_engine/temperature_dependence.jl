include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

# Interaction properties
d = 0       # Potential offset
l = 1       # Hamonic oscillator length
Φ0 = -5     # Potential magnitude

σs = [1 / 4, 1 / 2, 1, 2]   # Potential widths
# Oscillator properties
max_level = 60
single_basis = 0:max_level

nPts = 200
ωTs = range(1, 5, length = nPts)

for σ in σs
    y_min = 0
    y_max = 3 * σ
    ys = range(y_min, y_max, length = nPts)
    # Hamiltonian
    H_osc = diagm(single_basis .+ 1 / 2)
    Φ_mat = Matrix{Float64}([
        mode_interaction(n, m, l, d, σ) for n in single_basis, m in single_basis
    ])
    if !isfile(
        "data/piston_engine/temperature_dependence/Temperature_Dependence_Heatmap_$(σ).jld2",
    )
        function interaction_energy(y, ωT)
            H = Matrix{Float64}(H_osc + Φ0 * Φ_mat * exp(-y^2 / 2 / σ^2))
            interaction = Matrix{Float64}(Φ0 * Φ_mat * exp(-y^2 / 2 / σ^2))

            ρ_ = exp(-H ./ ωT)
            ρ_ = ρ_ ./ tr(ρ_)
            res = tr(ρ_ * interaction)
            return res
        end

        res = [interaction_energy(y, ωT) for y in ys, ωT in ωTs]
        save_object(
            "data/piston_engine/temperature_dependence/Temperature_Dependence_Heatmap_$(σ).jld2",
            (ys, ωTs, res),
        )
    end
end

## FIGURES
set_theme!(CF_theme)

data = readdir("data/piston_engine/temperature_dependence", join = true)

fig = Figure(size = (1200, 400))

supertitle = fig[1:2, 1]
Label(
    supertitle,
    "Piston-oscillator interaction",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

main_grid = fig[3:10, 1] = GridLayout()
data_grid = main_grid[1, 1:9] = GridLayout()
σ_lab = [L"\sigma = 1/4", L"\sigma = 1/2", L"\sigma = 1", L"\sigma = 2"]
axs = [
    Axis(data_grid[1, ii], xlabel = L"y / \sigma", ylabel = L"\omega_T", title = σ_lab[ii]) for ii in eachindex(data)
]
min_val = zeros(length(data))
for ii in eachindex(axs)

    r = load_object(data[ii])
    heatmap!(
        axs[ii],
        r[1] ./ σs[ii],
        r[2],
        r[3],
        colormap = CF_heat,
        colorrange = (-4.8, 0),
    )
    contour!(axs[ii], r[1] ./ σs[ii], r[2], r[3]; color = :black, levels = -4.8:0.2:0)
    min_val[ii] = minimum(r[3])
end

## ADD CYCLE

[hideydecorations!(axs[ii]) for ii in eachindex(data)[2:end]]

labs = ["(a)", "(b)", "(c)", "(d)"]
[
    text!(
        axs[ii],
        0.8,
        0.95,
        text = labs[ii],
        align = (:left, :top),
        space = :relative,
        fontsize = 36,
        font = :latex,
        color = :black,
    ) for ii in eachindex(labs)
]

cb = Colorbar(
    main_grid[1, 10],
    limits = (-4.8, 0),
    colormap = CF_heat,
    labelfont = :latex,
    ticklabelfont = :latex,
)

fig
save("Piston_interaction.pdf", fig)
