using RadiiPolynomial, LinearAlgebra
using GLMakie





# Step 1: Defining the problem

F(x, λ) = x^2 - λ

DλF(x, λ) = exact(-1)

DxF(x, λ) = exact(2) * x

F_pa(u, u̇, w) = [F(u[1], u[2])
                 sum(u̇ .* (u - w))]

DF_pa(u, u̇) = [DxF(u[1], u[2]) DλF(u[1], u[2])
               u̇[1]            u̇[2]]





# Step 2: Computing the approximate zero and inverse

arclength = sqrt(5) + 0.5 * log(2 + sqrt(5))

N = 2^5

arclength_grid = [0.5 * (1 - cospi(j/N)) * arclength for j ∈ 0:N]

u_grid = Vector{Vector{Float64}}(undef, N+1)
u̇_grid = Vector{Vector{Float64}}(undef, N+1)
A_grid = Vector{Matrix{Float64}}(undef, N+1)

# initialize

λ_bar = 1.0

x_init = 1.0
x_bar, converged = newton(x_init) do x
    return F(x, λ_bar), DxF(x, λ_bar)
end
converged || error("Newton failed")

u_bar = [x_bar, λ_bar]

u_grid[1] = u_bar

u̇_grid[1] = vec(nullspace([DxF(u_grid[1][1], u_grid[1][2]) DλF(u_grid[1][1], u_grid[1][2])]))
if u̇_grid[1][end] > 0 # decrease the parameter
    u̇_grid[1] .*= -1
end

A_grid[1] = inv(DF_pa(u_grid[1], u̇_grid[1]))

# run continuation scheme

for j ∈ 2:N+1
    δ = arclength_grid[j] - arclength_grid[j-1]

    predictor = u_grid[j-1] + δ * u̇_grid[j-1]

    u_barⱼ, convergedⱼ = newton(predictor) do u
        return F_pa(u, u̇_grid[j-1], predictor), DF_pa(u, u̇_grid[j-1])
    end
    convergedⱼ || error("Newton failed")

    u_grid[j] = u_barⱼ

    u̇_grid[j] = vec(nullspace([DxF(u_grid[j][1], u_grid[j][2]) DλF(u_grid[j][1], u_grid[j][2])]))
    if sum(u̇_grid[j-1] .* u̇_grid[j]) < 0 # keep the same direction
        u̇_grid[j] .*= -1
    end

    A_grid[j] = inv(DF_pa(u_grid[j], u̇_grid[j]))
end

# retrieve the Chebyshev interpolants

u_cheb = [interval(real(to_coef(getindex.(u_grid, i), Chebyshev(N)))) for i ∈ 1:2]

u̇_cheb = [interval(real(to_coef(getindex.(u̇_grid, i), Chebyshev(N)))) for i ∈ 1:2]

A_cheb = [interval(real(to_coef(getindex.(A_grid, i, j), Chebyshev(N)))) for i ∈ 1:2, j ∈ 1:2]





# Step 3: Estimating the bounds

#- Y bound

Y = norm(norm.(A_cheb * F_pa(u_cheb, u̇_cheb, u_cheb), 1), 1)

#- Z₁ bound

Z₁ = opnorm(norm.(exact(I(2)) - A_cheb * DF_pa(u_cheb, u̇_cheb), 1), 1)

#- Z₂ bound

r_star = Inf

Z₂ = exact(2) * opnorm(norm.(A_cheb, 1), 1)

#-

ie, proved = interval_of_existence(Y, Z₁, Z₂, r_star; verbose = true)





# Plot

fig_sqrt = Figure()
ax_sqrt = Axis(fig_sqrt[1,1]; aspect = DataAspect())

lines!(ax_sqrt, LinRange(0, 1.5, 501), λ ->  sqrt(λ);
    color = :red, linewidth = 1, linestyle = :dash, label = "square root")
lines!(ax_sqrt, LinRange(0, 1.5, 501), λ -> -sqrt(λ);
    color = :red, linewidth = 1, linestyle = :dash, label = "square root")

for i ∈ 1:N+1
    lines!(ax_sqrt, [Point2f(
        (-(r + u_grid[i][1]) * u̇_grid[i][1] + dot(u_grid[i], u̇_grid[i]))/u̇_grid[i][2],
           r + u_grid[i][1])
        for r ∈ LinRange(-0.1u̇_grid[i][2], 0.1u̇_grid[i][2], 2)]; color = :green, label = "intersecting planes")
end

lines!(ax_sqrt, [Point2f(mid(u_cheb[2])(s), mid(u_cheb[1])(s)) for s ∈ LinRange(-1, 1, 101)];
    color = :blue, linewidth = 2, label = "Chebyshev interpolant")
scatter!(ax_sqrt, Point2f.(reverse.(u_grid)); color = :blue)

axislegend(ax_sqrt; position = :rc, merge = true)

display(fig_sqrt)
