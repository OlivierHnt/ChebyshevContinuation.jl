using RadiiPolynomial, LinearAlgebra
using CairoMakie # for interactivity, use GLMakie instead





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

set_theme!(theme_latexfonts(); fontsize = 14)

col_blue, col_green, col_red = colorant"#4477AA", colorant"#228833", colorant"#EE6677" # Tol bright palette

fig_sqrt = Figure(; size = (550, 430)) # compact, true aspect: suited to the LaTeX scale option
ax_sqrt = Axis(fig_sqrt[1,1]; xlabel = L"\lambda", ylabel = L"x", aspect = DataAspect())

lines!(ax_sqrt, LinRange(0, 1.5, 501), λ ->  sqrt(λ);
    color = :gray30, linewidth = 1.5, linestyle = :dash, label = L"x^2 = \lambda")
lines!(ax_sqrt, LinRange(0, 1.5, 501), λ -> -sqrt(λ);
    color = :gray30, linewidth = 1.5, linestyle = :dash, label = L"x^2 = \lambda")

for j ∈ 1:N+1
    lines!(ax_sqrt, [Point2f(
        (-(r + u_grid[j][1]) * u̇_grid[j][1] + dot(u_grid[j], u̇_grid[j]))/u̇_grid[j][2],
           r + u_grid[j][1])
        for r ∈ LinRange(-0.1u̇_grid[j][2], 0.1u̇_grid[j][2], 2)];
        color = (col_red, 0.8), linewidth = 1.5, label = "intersecting planes")
end

lines!(ax_sqrt, [Point2f(mid(u_cheb[2])(s), mid(u_cheb[1])(s)) for s ∈ LinRange(-1, 1, 501)];
    color = col_blue, linewidth = 2.5, label = "Chebyshev interpolant")
scatter!(ax_sqrt, Point2f.(reverse.(u_grid)); color = col_blue, markersize = 7, label = "grid points")

#- the two components of the interpolant along the arclength
# the Chebyshev variable s ∈ [-1,1] relates to the arclength σ via σ = arclength * (1 - s) / 2

ax2_sqrt = Axis(fig_sqrt[1,2]; xlabel = L"\frac{L}{2}(s + 1)")

lines!(ax2_sqrt, [Point2f(arclength * (1 - s) / 2, mid(u_cheb[2])(s)) for s ∈ LinRange(-1, 1, 501)];
    color = col_green, linewidth = 2.5)
scatter!(ax2_sqrt, arclength_grid, getindex.(u_grid, 2); color = col_green, markersize = 7)

lines!(ax2_sqrt, [Point2f(arclength * (1 - s) / 2, mid(u_cheb[1])(s)) for s ∈ LinRange(-1, 1, 501)];
    color = col_blue, linewidth = 2.5)
scatter!(ax2_sqrt, arclength_grid, getindex.(u_grid, 1); color = col_blue, markersize = 7)

text!(ax2_sqrt, Point2f(2.45, 0.75); text = L"\lambda", color = col_green, fontsize = 18)
text!(ax2_sqrt, Point2f(2.45, -0.45); text = L"x", color = col_blue, fontsize = 18)

#-

Legend(fig_sqrt[2,1:2], ax_sqrt;
    merge = true, framevisible = false, orientation = :horizontal, nbanks = 2)

colsize!(fig_sqrt.layout, 1, Aspect(1, 0.68)) # so the DataAspect panel fills its column (data ratio 1.7:2.5)

display(fig_sqrt)

save(joinpath(@__DIR__, "square_root.pdf"), fig_sqrt)
