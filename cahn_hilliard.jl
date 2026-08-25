using RadiiPolynomial, LinearAlgebra
using GLMakie





# Step 1: Defining the problem

# Coons' parameterization

struct BoundaryCurves{T₁,T₂,T₃,T₄}
    γ₁ :: T₁
    γ₂ :: T₂
    γ₃ :: T₃
    γ₄ :: T₄
end

α(s) = (s + 1) / 2 # [-1,1] → [0,1]

θ(curves::BoundaryCurves, s₁, s₂) =
    θ_edge(s₁, s₂, curves.γ₁, curves.γ₃) +
    θ_edge(-s₂, s₁, curves.γ₄, curves.γ₂) -
    θ_corners(s₁, s₂; P₁ = curves.γ₁(-1), P₂ = curves.γ₁(1), P₃ = curves.γ₃(-1), P₄ = curves.γ₃(1))

θ_edge(s₁, s₂, γᵢ, γⱼ) = γᵢ(s₁) + α(s₂) * (γⱼ(-s₁) - γᵢ(s₁))

θ_corners(s₁, s₂; P₁, P₂, P₃, P₄) =
    P₁ + α(s₁) * (P₂ - P₁) + α(s₂) * (P₄ - P₁ + α(s₁) * (P₃ - P₄ - P₂ + P₁))

#- boundary curves with no pinch

γ_no_pinch₁(s) = [ 0.1 * (s - 3) / 4, 39.5 + 150 * (0.1 * (s - 3) / 4)^2]
γ_no_pinch₂(s) = [ 0.1 * s       / 2, 39.5 + 150 * (0.1 * s       / 2)^2]
γ_no_pinch₃(s) = [ 0.1 * (s + 3) / 4, 39.5 + 150 * (0.1 * (s + 3) / 4)^2]
γ_no_pinch₄(s) = [-0.1 * s, 41]

curves_no_pinch = BoundaryCurves(γ_no_pinch₁, γ_no_pinch₂, γ_no_pinch₃, γ_no_pinch₄)

#- boundary curves with pinch

γ_with_pinch₁(s) = [-0.1, 41]
γ_with_pinch₂(s) = [ 0.1 * s, 39.5 + 150 * (0.1 * s)^2]
γ_with_pinch₃(s) = [ 0.1, 41]
γ_with_pinch₄(s) = [-0.1 * s, 41]

curves_with_pinch = BoundaryCurves(γ_with_pinch₁, γ_with_pinch₂, γ_with_pinch₃, γ_with_pinch₄)

#

fig_curves = Figure(; size = (1200, 600))

N_plot = 2^6

ax1_curves = Axis(fig_curves[1,1]; title = "Boundary curves with no pinch")
scatter!(ax1_curves, vec([Point2f(θ(curves_no_pinch, -cospi(j₁/N_plot), -cospi(j₂/N_plot))) for j₁ ∈ 0:N_plot, j₂ ∈ 0:N_plot]))

ax2_curves = Axis(fig_curves[1,2]; title = "Boundary curves with pinch")
scatter!(ax2_curves, vec([Point2f(θ(curves_with_pinch,  -cospi(j₁/N_plot), -cospi(j₂/N_plot))) for j₁ ∈ 0:N_plot, j₂ ∈ 0:N_plot]))

display(fig_curves); sleep(0.1)

#

F(v, c, β) = Laplacian() * v + β * (v - v^3 - c)

DF(v, c, β) = Laplacian() + Multiplication(β * (exact(1) - exact(3) * v^2))





# Step 2: Computing the approximate zero and finite part of the inverse

for curves ∈ (curves_no_pinch, curves_with_pinch)

N₁, N₂ = 2^4, 2^4

θ_grid = [θ(curves, -cospi(j₁/N₁), -cospi(j₂/N₂)) for j₁ ∈ 0:N₁, j₂ ∈ 0:N₂]

v_grid = Matrix{Sequence}(undef, N₁+1, N₂+1)
A_finite_grid = Matrix{LinearOperator}(undef, N₁+1, N₂+1)

# initialize

c, β = θ_grid[1,1]

K = 20
v_init = Sequence(evensym(Fourier(K, π)), [-0.1022666446473428 ; 0 ; 0.044213651292169816 ; [rand()/5.6^k for k = 3:K]])
v_bar, converged = newton(v_init) do v
    return F(v, c, β), DF(v, c, β)
end
converged || error("Newton failed")

v_grid[1,1] = v_bar

Π = Projection(space(v_grid[1,1]))
A_finite_grid[1,1] = inv(Π * DF(v_grid[1,1], θ_grid[1,1]...) * Π)

# run continuation scheme

for j₂ ∈ 1:N₂+1, j₁ ∈ 1:N₁+1
    if !(j₁ == j₂ == 1)
        cⱼ, βⱼ = θ_grid[j₁,j₂]

        predictor = j₁ == 1 ? v_grid[1,j₂-1] : v_grid[j₁-1,j₂]

        v_barⱼ, convergedⱼ = newton(predictor) do v
            return F(v, cⱼ, βⱼ), DF(v, cⱼ, βⱼ)
        end
        convergedⱼ || error("Newton failed")

        v_grid[j₁,j₂] = v_barⱼ

        A_finite_grid[j₁,j₂] = inv(Π * DF(v_grid[j₁,j₂], cⱼ, βⱼ) * Π)
    end
end

maximum(vᵢ -> log10(abs(vᵢ[end])), v_grid) # with K = 20, error depends (roughly) on the interpolation order

# retrieve the Chebyshev interpolants

c_cheb = interval(real(to_coef(getindex.(reverse(θ_grid; dims=(1,2)), 1), Chebyshev(N₁) ⊗ Chebyshev(N₂))))
β_cheb = interval(real(to_coef(getindex.(reverse(θ_grid; dims=(1,2)), 2), Chebyshev(N₁) ⊗ Chebyshev(N₂))))

v_cheb = interval(real(to_coef(reverse(v_grid; dims=(1,2)), Chebyshev(N₁) ⊗ Chebyshev(N₂))))

A_finite_cheb = interval(real(to_coef(reverse(A_finite_grid; dims=(1,2)), Chebyshev(N₁) ⊗ Chebyshev(N₂))))





# Step 4: Estimating the bounds

# utilitary

struct PseudoInverseLaplacian <: AbstractDiagonalOperator end
RadiiPolynomial.getcoefficient(::PseudoInverseLaplacian, (codom, i)::Tuple{SymmetricSpace{<:Fourier},Integer}, (dom, j)::Tuple{SymmetricSpace{<:Fourier},Integer}) =
    (i == j) & !(i == j == 0) ? inv(-(frequency(dom) * exact(i))^2) : zero(frequency(dom))

Δ⁻¹ = PseudoInverseLaplacian()

# sample on the largest grid: Y bound requires 4N

N₁_Y, N₂_Y = 4N₁, 4N₂

c_grid_Y = to_grid(c_cheb, (N₁_Y+1, N₂_Y+1));
β_grid_Y = to_grid(β_cheb, (N₁_Y+1, N₂_Y+1));

v_grid_Y = to_grid(v_cheb, (N₁_Y+1, N₂_Y+1));

A_finite_grid_Y = to_grid(A_finite_cheb, (N₁_Y+1, N₂_Y+1));

A_grid_Y = A_finite_grid_Y .+ Δ⁻¹ * (interval(I) - interval(Π))

#- Y bound

Y_grid = A_grid_Y .* F.(v_grid_Y, c_grid_Y, β_grid_Y)
Y = norm(to_coef(Y_grid, Chebyshev(N₁_Y) ⊗ Chebyshev(N₂_Y)), Ell1())

#- Z₁ bound

Π_3Kp1 = interval(Projection(evensym(Fourier(3K+1, π))))
Z₁_grid = Π_3Kp1 .- A_grid_Y .* (DF.(v_grid_Y, c_grid_Y, β_grid_Y) .* Π_3Kp1)
Z₁ = opnorm.(to_coef(Z₁_grid, Chebyshev(N₁_Y) ⊗ Chebyshev(N₂_Y)), Ell1())

#- Z₂ bound

r_star = 10sup(Y)

normA = max(opnorm(A_finite_cheb, Ell1()), interval(1)/interval(π)^2)

Z₂ = exact(3) * norm(β_cheb, Ell1()) * normA * (exact(2) * norm(v_cheb, Ell1()) + exact(r_star))

#-

ie, proved = interval_of_existence(Y, Z₁, Z₂, r_star; verbose = true)





# Plot

fig_ch = Figure(; size = (1200, 600))

vals_ch = [norm(v_grid[j₁,j₂], 2) for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
cmap_ch = cgrad(:Sunset2, length(vals_ch); categorical = true);

#-

ax1_ch = Axis3(fig_ch[1,1]; viewmode = :free, aspect = :equal,
    xlabelsize = 28, ylabelsize = 28, zlabelsize = 28,
    zlabelrotation = 0, zlabeloffset = 75,
    xlabel = L"s_1", ylabel = L"s_2", zlabel = L"c")

surface!(ax1_ch,
    [-cospi(j₁/N₁)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [-cospi(j₂/N₂)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [θ_grid[j₁,j₂][1] for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
    colormap = cmap_ch, color = vals_ch, overdraw = false)

wireframe!(ax1_ch,
    [-cospi(j₁/N₁)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [-cospi(j₂/N₂)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [θ_grid[j₁,j₂][1] for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
    color = :black, linewidth = 1, overdraw = false, transparency = true)

#-

ax2_ch = Axis3(fig_ch[1,2]; viewmode = :free, aspect = :equal,
    xlabelsize = 28, ylabelsize = 28, zlabelsize = 28,
    zlabelrotation = 0, zlabeloffset = 75,
    xlabel = L"s_1", ylabel = L"s_2", zlabel = L"\beta")

surface!(ax2_ch,
    [-cospi(j₁/N₁)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [-cospi(j₂/N₂)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [θ_grid[j₁,j₂][2] for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
    colormap = cmap_ch, color = vals_ch, overdraw = false)

wireframe!(ax2_ch,
    [-cospi(j₁/N₁)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [-cospi(j₂/N₂)    for j₁ ∈ 0:N₁,   j₂ ∈ 0:N₂],
    [θ_grid[j₁,j₂][2] for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
    color = :black, linewidth = 1, overdraw = false, transparency = true)

#-

ax3_ch = Axis3(fig_ch[2,1:2]; viewmode = :free, aspect = :equal,
    xlabelsize = 28, ylabelsize = 28, zlabelsize = 28,
    zlabelrotation = 0, zlabeloffset = 75,
    xlabel = L"c", ylabel = L"\beta", zlabel = L"||u||")

surface!(ax3_ch,
    [θ_grid[j₁,j₂][1]       for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1],
    [θ_grid[j₁,j₂][2]       for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1],
    [norm(v_grid[j₁,j₂], 2) for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
    colormap = cmap_ch, color = vals_ch, overdraw = false)

wireframe!(ax3_ch,
    [θ_grid[j₁,j₂][1]       for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1],
    [θ_grid[j₁,j₂][2]       for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1],
    [norm(v_grid[j₁,j₂], 2) for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1];
    color = :black, linewidth = 1, overdraw = false, transparency = true)

Label(fig_ch[0,:], curves == curves_no_pinch ? "No pinch" : "With pinch", font = :bold)

display(GLMakie.Screen(), fig_ch)
end
