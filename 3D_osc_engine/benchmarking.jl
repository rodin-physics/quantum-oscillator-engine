include("../main.jl")
include("../plotting.jl")
using JLD2
using ProgressMeter

## Parameters
T = 5
max_level = T * 20
single_basis = 0:max_level

τ_max = 5           # Max evolution time
δτ_factors = 3:16

αs = [5, 25]        # Values of αs (either fixed or reached in the variable case)

# Gas Hamiltonian
H0_gas = diagm(single_basis .+ 1 / 2)
H_scatter_gas = diagm(1 => sqrt.(1:max_level))^2 + diagm(-1 => sqrt.(1:max_level))^2

function H_gas(α)
    return sparse(H0_gas * (1 + α / 2) + α / 4 * H_scatter_gas)
end

# Initial state
gas_state = diagm(exp.(-(single_basis .+ 1 / 2) ./ T))
gas_state = gas_state ./ tr(gas_state)

## FIXED α

println("Starting FIXED calculation")
for α in αs
    if !isfile("data/3D_Osc_Engine/benchmarking/Fixed_α_RK_Error_T$(T)_α$(α).jld2")
        H_fixed = H_gas(α)
        U_analytic = exp(-2im * π * τ_max * Matrix(H_fixed))
        gas_analytic = U_analytic * gas_state * U_analytic'

        ϵ = 1 / max_level / (2 * π) / (1 + α / 2)
        tr_dist = zeros(length(δτ_factors))
        pr = Progress(length(δτ_factors))

        # Calculate the final state for the smallest step size
        δτ = ϵ / δτ_factors[end]
        nSteps = floor(Int, τ_max / δτ)
        U_numeric = diagm(Vector{ComplexF64}(ones(max_level + 1)))

        @showprogress for ii = 0:(nSteps-1)
            U_numeric = RKstep(t -> H_fixed, U_numeric, ii * δτ, δτ)
        end
        U_numeric = RKstep(t -> H_fixed, U_numeric, nSteps * δτ, τ_max - nSteps * δτ)
        gas_finest = U_numeric * gas_state * U_numeric'
        next!(pr)

        for jj = 1:(length(δτ_factors)-1)
            δτ = ϵ / δτ_factors[jj]
            nSteps = floor(Int, τ_max / δτ)
            U_numeric = diagm(Vector{ComplexF64}(ones(max_level + 1)))

            @showprogress for ii = 0:(nSteps-1)
                U_numeric = RKstep(t -> H_fixed, U_numeric, ii * δτ, δτ)
                GC.safepoint()
            end
            U_numeric = RKstep(t -> H_fixed, U_numeric, nSteps * δτ, τ_max - nSteps * δτ)
            gas_numeric = U_numeric * gas_state * U_numeric'
            tr_dist[jj] = real(trace_distance(gas_finest, gas_numeric))
            next!(pr)
            GC.safepoint()
            save_object(
                "data/3D_Osc_Engine/benchmarking/Fixed_α_RK_Error_T$(T)_α$(α).jld2",
                (δτ_factors, tr_dist, T, max_level, α),
            )
        end
    end
end
## VARIABLE α

println("Starting VARIABLE calculation")
for α_max in αs
    function α(t)
        return (t / τ_max * α_max)
    end

    if !isfile("data/3D_Osc_Engine/benchmarking/Variable_α_RK_Error_T$(T)_α$(α_max).jld2")
        ϵ = 1 / max_level / (2 * π) / (1 + α_max / 2)
        tr_dist = zeros(length(δτ_factors))
        pr = Progress(length(δτ_factors))

        # Calculate the final state for the smallest step size
        δτ = ϵ / δτ_factors[end]
        nSteps = floor(Int, τ_max / δτ)
        U_numeric = diagm(Vector{ComplexF64}(ones(max_level + 1)))

        @showprogress for ii = 0:(nSteps-1)
            U_numeric = RKstep(t -> H_gas(α(t)), U_numeric, ii * δτ, δτ)
        end
        U_numeric = RKstep(t -> H_gas(α(t)), U_numeric, nSteps * δτ, τ_max - nSteps * δτ)
        gas_finest = U_numeric * gas_state * U_numeric'
        next!(pr)
        for jj = 1:(length(δτ_factors)-1)
            δτ = ϵ / δτ_factors[jj]
            nSteps = floor(Int, τ_max / δτ)
            U_numeric = diagm(Vector{ComplexF64}(ones(max_level + 1)))

            @showprogress for ii = 0:(nSteps-1)
                U_numeric = RKstep(t -> H_gas(α(t)), U_numeric, ii * δτ, δτ)
                GC.safepoint()
            end
            U_numeric =
                RKstep(t -> H_gas(α(t)), U_numeric, nSteps * δτ, τ_max - nSteps * δτ)
            gas_numeric = U_numeric * gas_state * U_numeric'
            tr_dist[jj] = real(trace_distance(gas_finest, gas_numeric))
            next!(pr)
            GC.safepoint()
            save_object(
                "data/3D_Osc_Engine/benchmarking/Variable_α_RK_Error_T$(T)_α$(α_max).jld2",
                (δτ_factors, tr_dist, T, max_level, α_max),
            )
        end
    end
end

## FIGURE
set_theme!(CF_theme)
fig = Figure(resolution = (1200, 600))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Runge-Kutta Error",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 42,
    valign = :center,
)

main_grid = fig[2:10, 1] = GridLayout()
res_grid = main_grid[1, 1] = GridLayout()

ax = Axis(
    res_grid[1, 1],
    yscale = log10,
    xaxisposition = :bottom,
    xticks = 3:3:15,
    xlabel = L"\varepsilon / \delta\tau",
    ylabel = "Error",
)

f5 = load_object("data/3D_Osc_Engine/benchmarking/Fixed_α_RK_Error_T5_α5.jld2")
f25 = load_object("data/3D_Osc_Engine/benchmarking/Fixed_α_RK_Error_T5_α25.jld2")
v5 = load_object("data/3D_Osc_Engine/benchmarking/Variable_α_RK_Error_T5_α5.jld2")
v25 = load_object("data/3D_Osc_Engine/benchmarking/Variable_α_RK_Error_T5_α25.jld2")
marker_size = 20
scatter!(
    ax,
    f5[1][1:end-1],
    f5[2][1:end-1],
    color = CF_vermillion,
    marker = :cross,
    markersize = marker_size,
    label = L"\alpha_\mathrm{fixed} = 5",
)
scatter!(
    ax,
    f25[1][1:end-1],
    f25[2][1:end-1],
    color = CF_vermillion,
    markersize = marker_size,
    label = L"\alpha_\mathrm{fixed} = 25",
)
scatter!(
    ax,
    v5[1][1:end-1],
    v5[2][1:end-1],
    color = CF_sky,
    marker = :cross,
    markersize = marker_size,
    label = L"\alpha_\mathrm{max} = 5",
)
scatter!(
    ax,
    v25[1][1:end-1],
    v25[2][1:end-1],
    color = CF_sky,
    markersize = marker_size,
    label = L"\alpha_\mathrm{max} = 25",
)

ylims!(ax, (1e-13, 1e-5))
xlims!(ax, (2, 15.5))

axislegend(ax, framevisible = false)
fig

save("Error.pdf", fig)
