include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

# Interaction properties
d = 1       # Potential offset
σ = 1       # Potential width
Φ0 = 1      # Potential magnitude

# Oscillator properties
max_level = 40
ω = 1       # Frequency
single_basis = 0:max_level
α = ω^2 - 1

# Hamiltonian
H0_gas = diagm(single_basis .+ 1 / 2)
H_scatter_gas = diagm(1 => sqrt.(1:max_level))^2 + diagm(-1 => sqrt.(1:max_level))^2
H_osc = H0_gas * (1 + α / 2) + α / 4 * H_scatter_gas
id = diagm(ones(length(single_basis)))

Φ_g = [mode_interaction(n, m, 1, d, σ) for n in single_basis, m in single_basis]
Φ_mat = Matrix{Float64}(kron(Φ_g, Φ_g))

H_total = kron(H_osc, id) + kron(id, H_osc) + Φ0 .* Φ_mat

# Oscillator states
Th = 5      # Hot temperature
Tc = 1 / 10   # Cold temperature

Gibbs_hot = exp(-H_osc ./ Th)
Gibbs_hot = Gibbs_hot ./ tr(Gibbs_hot)

Gibbs_cold = exp(-H_osc ./ Tc)
Gibbs_cold = Gibbs_cold ./ tr(Gibbs_cold)

# Evolution properties
δτ = 1e-1
nPts = 600
U = unitary_evolution(H_total, δτ) |> sparse

prob = Vector{Vector{Float64}}(undef, nPts)

ρ_tot = kron(Gibbs_cold, Gibbs_hot)
prob[1] = diag(ρ_tot)

if !isfile("data/3D_Osc_Engine/animation/animation_data.jld2")
    @showprogress for ii = 2:nPts
        ρ_tot = U * ρ_tot * U'
        prob[ii] = real.(diag(ρ_tot))
    end
    save_object("data/3D_Osc_Engine/animation/animation_data.jld2", prob)
end

## Creating frames
data = load_object("data/3D_Osc_Engine/animation/animation_data.jld2")

set_theme!(CF_theme)

for ii in eachindex(data)
    fig = Figure(size = (800, 800))

    ax = Axis(
        fig[1, 1],
        xlabel = "Hot oscillator levels",
        ylabel = "Cold oscillator levels",
        title = "Thermal equilibration of two SHO's",
        aspect = DataAspect(),
    )

    heatmap!(ax, reshape(data[ii], max_level + 1, :), colormap = CF_heat)
    xlims!(ax, (0, 20))
    ylims!(ax, (0, 20))
    fig
    save("data/3D_Osc_Engine/animation/frame_$(ii).png", fig)
    # save("data/3D_Osc_Engine/animation/frame_$(lpad(ii, 3, '0')).png", fig)

end
