using RadiiPolynomial, LinearAlgebra
using CairoMakie # for interactivity, use GLMakie instead





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

F(v, c, β) = Laplacian() * v + β * (v - v^3 - c)

DF(v, c, β) = Laplacian() + Multiplication(β * (exact(1) - exact(3) * v^2))





# Step 2: Computing the approximate zero and finite part of the inverse

patch_data = [] # per-case data for the figures drawn after the loop

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





# collect the data of this parameterization for the figures drawn after the loop

push!(patch_data, (;
    s₁_grid = [-cospi(j₁/N₁) for j₁ ∈ 0:N₁, j₂ ∈ 0:N₂],
    s₂_grid = [-cospi(j₂/N₂) for j₁ ∈ 0:N₁, j₂ ∈ 0:N₂],
    c_grid = getindex.(θ_grid, 1),
    β_grid = getindex.(θ_grid, 2),
    vals = [norm(v_grid[j₁,j₂], 2) for j₁ ∈ 1:N₁+1, j₂ ∈ 1:N₂+1]))
end





# Plot: the solution manifold (identical for both parameterizations, hence no wireframe)

set_theme!(theme_latexfonts(); fontsize = 14)

fig_manifold = Figure(; size = (500, 420))

ax_manifold = Axis3(fig_manifold[1,1]; protrusions = (70, 10, 35, 5),
    zlabelrotation = 0, zlabeloffset = 65,
    xticks = [-0.1, 0, 0.1],
    xlabel = L"c", ylabel = L"\beta", zlabel = L"\Vert v \Vert_2")

surface!(ax_manifold, patch_data[1].c_grid, patch_data[1].β_grid, patch_data[1].vals;
    colormap = :batlow, color = patch_data[1].vals, colorrange = extrema(patch_data[1].vals),
    rasterize = 4) # embed as bitmap: avoids white polygon seams and rendering glitches in PDF viewers

display(fig_manifold)

save(joinpath(@__DIR__, "cahn_hilliard_manifold.pdf"), fig_manifold)





# Plot: the parameterization of (c, β) for each choice of boundary curves
# A, B: no pinch — C, D: with pinch

fig_param = Figure(; size = (605, 540), figure_padding = (8, 8, 8, 4)) # 605px ≈ 16cm: matches \textwidth at natural size

crange = extrema(vcat(vec(patch_data[1].vals), vec(patch_data[2].vals)))

for (r, data) ∈ enumerate(patch_data)
    ax_c = Axis3(fig_param[r,1]; protrusions = (70, 10, 25, 25),
        zlabelrotation = 0, zlabeloffset = 55,
        xticks = [-1, 0, 1], yticks = [-1, 0, 1],
        xlabel = L"s_1", ylabel = L"s_2", zlabel = L"c")

    surface!(ax_c, data.s₁_grid, data.s₂_grid, data.c_grid;
        colormap = :batlow, color = data.vals, colorrange = crange,
        rasterize = 4) # embed as bitmap: avoids white polygon seams and rendering glitches in PDF viewers

    wireframe!(ax_c, data.s₁_grid, data.s₂_grid, data.c_grid;
        color = (:black, 0.25), linewidth = 0.6)

    ax_β = Axis3(fig_param[r,2]; protrusions = (10, 70, 25, 25),
        azimuth = 1.6π, # rotated so the valley of the β surface is visible
        zlabelrotation = 0, zlabeloffset = 55,
        xticks = [-1, 0, 1], yticks = [-1, 0, 1],
        xlabel = L"s_1", ylabel = L"s_2", zlabel = L"\beta")

    surface!(ax_β, data.s₁_grid, data.s₂_grid, data.β_grid;
        colormap = :batlow, color = data.vals, colorrange = crange,
        rasterize = 4) # embed as bitmap: avoids white polygon seams and rendering glitches in PDF viewers

    wireframe!(ax_β, data.s₁_grid, data.s₂_grid, data.β_grid;
        color = (:black, 0.25), linewidth = 0.6)
end

for (panel_label, (r, c)) ∈ zip(["A", "B", "C", "D"], [(1,1), (1,2), (2,1), (2,2)])
    Label(fig_param[r, c, Top()], panel_label; font = :bold, fontsize = 18, halign = :left)
end

rowgap!(fig_param.layout, 5)

display(fig_param)

save(joinpath(@__DIR__, "cahn_hilliard_parameterization.pdf"), fig_param)





# Plot: local density of the Chebyshev nodes under the Coons parameterization
#
# Each grid is colored by the local node density (reciprocal of the local cell
# area, i.e. the Jacobian of the parameterization) on a log scale: dark regions
# indicate where the Chebyshev nodes accumulate.

N_nodes = 2^8

cheb_nodes = [-cospi(j/N_nodes) for j ∈ 0:N_nodes]

square_pts = [Point2f(s₁, s₂) for s₁ ∈ cheb_nodes, s₂ ∈ cheb_nodes]
no_pinch_pts = [Point2f(θ(curves_no_pinch, s₁, s₂)) for s₁ ∈ cheb_nodes, s₂ ∈ cheb_nodes]
with_pinch_pts = [Point2f(θ(curves_with_pinch, s₁, s₂)) for s₁ ∈ cheb_nodes, s₂ ∈ cheb_nodes]

function log10_density(P)
    n₁, n₂ = size(P)
    d = zeros(Float32, n₁, n₂)
    for j₂ ∈ 1:n₂, j₁ ∈ 1:n₁
        i⁺, i⁻ = min(j₁+1, n₁), max(j₁-1, 1)
        k⁺, k⁻ = min(j₂+1, n₂), max(j₂-1, 1)
        t₁ = (P[i⁺,j₂] - P[i⁻,j₂]) / (i⁺ - i⁻)
        t₂ = (P[j₁,k⁺] - P[j₁,k⁻]) / (k⁺ - k⁻)
        cell_area = abs(t₁[1] * t₂[2] - t₁[2] * t₂[1])
        d[j₁,j₂] = -log10(max(cell_area, 1e-12))
    end
    return d
end

function density_mesh!(ax, P; colormap, span = 1.5) # span: decades of density above the sparsest region
    n₁, n₂ = size(P)
    node_idx(j₁, j₂) = j₁ + (j₂ - 1) * n₁
    faces = Matrix{Int}(undef, 2 * (n₁ - 1) * (n₂ - 1), 3)
    f = 0
    for j₂ ∈ 1:n₂-1, j₁ ∈ 1:n₁-1
        a, b, c, d = node_idx(j₁, j₂), node_idx(j₁+1, j₂), node_idx(j₁+1, j₂+1), node_idx(j₁, j₂+1)
        faces[f += 1, :] .= (a, b, c)
        faces[f += 1, :] .= (a, c, d)
    end
    ld = log10_density(P)
    dmin = minimum(ld)
    mesh!(ax, vec(P), faces;
        color = vec(ld), colormap, colorrange = (dmin, dmin + span), shading = NoShading,
        rasterize = 4) # embed as high-res bitmap: keeps the PDF small, axes/text stay vector
end

set_theme!(theme_latexfonts(); fontsize = 14)

col_blue, col_green, col_red = colorant"#4477AA", colorant"#228833", colorant"#EE6677" # Tol bright palette

cmap_density = cgrad([colorant"#ffffff", col_blue, colorant"#131f30"])

fig_coons = Figure(; size = (605, 230)) # 605px ≈ 16cm: matches \textwidth at natural size

#-

corners_square = Point2f.([(-1, -1), (1, -1), (1, 1), (-1, 1)])

ax1_coons = Axis(fig_coons[1,1]; xlabel = L"s_1", ylabel = L"s_2",
    xticks = [-1, 0, 1], yticks = [-1, 0, 1])

colsize!(fig_coons.layout, 1, Aspect(1, 1)) # square cell: 1:1 ratio for the Chebyshev grid, its height sets the row

density_mesh!(ax1_coons, square_pts; colormap = cmap_density)

scatter!(ax1_coons, corners_square; markersize = 8, color = :black, strokecolor = :white, strokewidth = 1)
text!(ax1_coons, corners_square .+ Point2f.([(-0.06, -0.06), (0.06, -0.06), (0.06, 0.06), (-0.06, 0.06)]);
    text = [L"P_1", L"P_2", L"P_3", L"P_4"],
    align = [(:right, :top), (:left, :top), (:left, :bottom), (:right, :bottom)])

xlims!(ax1_coons, -1.5, 1.5)
ylims!(ax1_coons, -1.5, 1.5)

#-

corners_no_pinch = Point2f.([γ_no_pinch₁(-1), γ_no_pinch₁(1), γ_no_pinch₃(-1), γ_no_pinch₃(1)])

ax2_coons = Axis(fig_coons[1,2]; xlabel = L"c", ylabel = L"\beta", xticks = [-0.1, 0, 0.1])

density_mesh!(ax2_coons, no_pinch_pts; colormap = cmap_density)

scatter!(ax2_coons, corners_no_pinch; markersize = 8, color = :black, strokecolor = :white, strokewidth = 1)
text!(ax2_coons, corners_no_pinch .+ Point2f.([(-0.008, 0.04), (-0.008, -0.04), (0.008, -0.04), (0.008, 0.04)]);
    text = [L"P_1", L"P_2", L"P_3", L"P_4"],
    align = [(:right, :bottom), (:right, :top), (:left, :top), (:left, :bottom)])

xlims!(ax2_coons, -0.165, 0.165)
ylims!(ax2_coons, 39.35, 41.3)

#-

corners_with_pinch = Point2f.([γ_with_pinch₁(-1), γ_with_pinch₃(-1)])

ax3_coons = Axis(fig_coons[1,3]; xlabel = L"c", ylabel = L"\beta", xticks = [-0.1, 0, 0.1])

density_mesh!(ax3_coons, with_pinch_pts; colormap = cmap_density)

scatter!(ax3_coons, corners_with_pinch; markersize = 8, color = :black, strokecolor = :white, strokewidth = 1)
text!(ax3_coons, corners_with_pinch .+ Point2f.([(0.018, 0.04), (-0.018, 0.04)]);
    text = [L"P_1 = P_2", L"P_3 = P_4"],
    align = [(:center, :bottom), (:center, :bottom)])

xlims!(ax3_coons, -0.165, 0.165)
ylims!(ax3_coons, 39.35, 41.3)

#-

display(fig_coons)

save(joinpath(@__DIR__, "cahn_hilliard_coons.pdf"), fig_coons)
