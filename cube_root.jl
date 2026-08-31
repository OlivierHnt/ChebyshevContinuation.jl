using RadiiPolynomial
using CairoMakie # for interactivity, use GLMakie instead





# Step 1: Defining the problem

F(x, λ) = x^3 - λ - exact(2)

DF(x) = exact(3) * x^2





# Step 2: Computing the approximate zero and inverse

N = 2^3

λ_grid = [-cospi(j/N) for j ∈ 0:N]

x_grid = Vector{Float64}(undef, N+1)
A_grid = Vector{Float64}(undef, N+1)

# initialize

x_bar = cbrt(λ_grid[1] + 2)

x_grid[1] = x_bar

A_grid[1] = inv(DF(x_grid[1]))

# run continuation scheme

for j ∈ 2:N+1
    λ = λ_grid[j]

    predictor = x_grid[j-1]

    x_barⱼ, convergedⱼ = newton(predictor) do x
        return F(x, λ), DF(x)
    end
    convergedⱼ || error("Newton failed")

    x_grid[j] = x_barⱼ

    A_grid[j] = inv(DF(x_grid[j]))
end

# retrieve the Chebyshev interpolants

λ_cheb = interval(Sequence(Chebyshev(1), [0.0, 0.5]))

x_cheb = interval(real(to_coef(reverse(x_grid), Chebyshev(N))))

A_cheb = interval(real(to_coef(reverse(A_grid), Chebyshev(N))))





# Step 3: Estimating the bounds

#- Y bound

Y = norm(A_cheb * F(x_cheb, λ_cheb), 1)

#- Z₁ bound

Z₁ = norm(exact(1) - A_cheb * DF(x_cheb), 1)

#- Z₂ bound

r_star = 1e-6 # 10sup(Y)

Z₂ = exact(6) * norm(A_cheb, 1) * (norm(x_cheb, 1) + exact(r_star))

#-

ie, proved = interval_of_existence(Y, Z₁, Z₂, r_star; verbose = true)





# Plot

set_theme!(theme_latexfonts(); fontsize = 14)

col_blue, col_green, col_red = colorant"#4477AA", colorant"#228833", colorant"#EE6677" # Tol bright palette

fig_cbrt = Figure(; size = (340, 265)) # compact, true aspect: suited to wrapped text and the LaTeX scale option
ax_cbrt = Axis(fig_cbrt[1,1]; xlabel = L"\lambda", ylabel = L"x", aspect = DataAspect())

lines!(ax_cbrt, LinRange(-3, 1.5, 501), λ -> cbrt(λ + 2);
    color = :gray30, linewidth = 1.5, linestyle = :dash, label = L"x^3 = \lambda + 2")

lines!(ax_cbrt, LinRange(-1, 1, 501), λ -> mid(x_cheb)(λ);
    color = col_blue, linewidth = 2.5, label = "Chebyshev interpolant")
scatter!(ax_cbrt, λ_grid, x_grid; color = col_blue, markersize = 7, label = "grid points")

leg_cbrt = fig_cbrt[2,1] = GridLayout()
Legend(leg_cbrt[1,1],
    [LineElement(; color = :gray30, linewidth = 1.5, linestyle = :dash)],
    [L"x^3 = \lambda + 2"]; framevisible = false, tellheight = true, tellwidth = false, halign = :right)
Legend(leg_cbrt[1,2],
    [LineElement(; color = col_blue, linewidth = 2.5),
     MarkerElement(; color = col_blue, marker = :circle, markersize = 7)],
    ["Chebyshev interpolant", "grid points"]; framevisible = false, tellheight = true, tellwidth = false, halign = :left)
colsize!(leg_cbrt, 1, Auto(0.35))
colsize!(leg_cbrt, 2, Auto(0.65))

display(fig_cbrt)

save(joinpath(@__DIR__, "cube_root.pdf"), fig_cbrt)
