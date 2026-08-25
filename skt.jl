using RadiiPolynomial, LinearAlgebra
using GLMakie

import JLD2



# Step 1: Defining the problem

const d₁₁, d₁₂, d₂₁, d₂₂ = exact(0), exact(3), exact(0), exact(0)
const r₁, a₁, b₁ = exact(5), exact(3), exact(1)
const r₂, b₂, a₂ = exact(2), exact(1), exact(3)

Φ(v, λ) = [(λ + d₁₁*v[1] + d₁₂*v[2]) * v[1],
           (λ + d₂₁*v[1] + d₂₂*v[2]) * v[2]]

DvΦ(v, λ) = [exact(2)*d₁₁*v[1]+d₁₂*v[2]+λ                      d₁₂*v[1]
                               d₂₁*v[2]      exact(2)*d₂₂*v[2]+d₂₁*v[1]+λ]

DλΦ(v) = v

R(v) = [(r₁ - a₁*v[1] - b₁*v[2]) * v[1],
        (r₂ - b₂*v[1] - a₂*v[2]) * v[2]]

DR(v) = [r₁-exact(2)*a₁*v[1]-b₁*v[2]                       -b₁*v[1]
                            -b₂*v[2]    r₂-exact(2)*a₂*v[2]-b₂*v[1]]

F(v, λ)   = Laplacian() .* Φ(v, λ) + R(v)

DvF(v, λ) = Laplacian() .* Multiplication.(DvΦ(v, λ)) + Multiplication.(DR(v))

DλF(v)    = LinearOperator.(Laplacian() .* DλΦ(v))

F_pa(u, u̇, w) = [F(u[1:2], u[3][1]) ; adjoint(u̇) * (u - w)]

DF_pa(u, u̇) = [DvF(u[1:2], u[3][1])   DλF(u[1:2])
               adjoint(u̇[1:2])        LinearOperator(u̇[3])]

#--

function tangent(u)
    ub = unpack(u)
    Πseq2 = Projection(space(u)[1:2])
    Π = Projection(space(u))
    D = Πseq2 * [DvF(ub[1:2], ub[3][1]) DλF(ub[1:2])] * Π
    return Sequence(space(u), vec(nullspace(real.(coefficients(D)))))
end

function approx_A(u, u̇, Π_K_A, Π_2K_A)
    DF_ = DF_pa(unpack(u), unpack(u̇))
    return real(Π_K_A * inv(Π_2K_A * DF_ * Π_2K_A) * Π_K_A)
end

#--



# Step 2: Computing the approximate branch piece and inverse (floating-point)

fig_skt = Figure()

ax_skt = Axis(fig_skt[1,1])

for (iter, name) ∈ enumerate(["u_bar1a", "u_bar1b", "u_bar1c",
                              "u_bar2a", "u_bar2b", "u_bar2c", "u_bar2d",
                              "u_bar3a", "u_bar3b", "u_bar3c", "u_bar3d",
                              "u_bar4a", "u_bar4b", "u_bar4c"])

arclength = Dict("u_bar1a" => 0.38, "u_bar1b" => 0.19, "u_bar1c" => 0.26,
                 "u_bar2a" => 0.40, "u_bar2b" => 0.10, "u_bar2c" => 0.10, "u_bar2d" => 0.32,
                 "u_bar3a" => 0.20, "u_bar3b" => 0.10, "u_bar3c" => 0.10, "u_bar3d" => 0.52,
                 "u_bar4a" => 0.20, "u_bar4b" => 0.20, "u_bar4c" => 0.40)[name]

N = Dict("u_bar1a" => 16, "u_bar1b" => 16, "u_bar1c" => 24,
         "u_bar2a" => 16, "u_bar2b" => 16, "u_bar2c" => 20, "u_bar2d" => 24,
         "u_bar3a" => 16, "u_bar3b" => 16, "u_bar3c" => 16, "u_bar3d" => 24,
         "u_bar4a" => 16, "u_bar4b" => 24, "u_bar4c" => 16)[name]

arclength_grid = [0.5 * (1 - cospi(j/N)) * arclength for j ∈ 0:N]

u_grid = Vector{Sequence}(undef, N+1)
u̇_grid = Vector{Sequence}(undef, N+1)
A_finite_grid = Vector{LinearOperator}(undef, N+1)
W_grid = Vector{LinearOperator}(undef, N+1)

# initialize

K = 80

data = JLD2.load(joinpath(@__DIR__, "skt_data.jld2"))

λ_bar = data[name][end]

# Kdata = length(data[name][1:end-1]) ÷ 2 - 1
# v_init = Projection(evensym(Fourier(K, π)) × evensym(Fourier(K, π))) * Sequence(evensym(Fourier(Kdata, π)) × evensym(Fourier(Kdata, π)), data[name][1:end-1])
v_init = Sequence(evensym(Fourier(K, π)) × evensym(Fourier(K, π)), data[name][1:end-1])
v_bar, converged = newton(v_init) do v
    return F(unpack(v), λ_bar), DvF(unpack(v), λ_bar)
end
converged || error("Newton failed")

u_bar = Sequence(evensym(Fourier(K, π)) × evensym(Fourier(K, π)) × ScalarSpace(),
    [coefficients(v_bar) ; λ_bar])

u_grid[1] = u_bar

direction = Dict("u_bar1a" => -1.0, "u_bar1b" => -1.0, "u_bar1c" => -1.0,
                 "u_bar2a" => -1.0, "u_bar2b" => -1.0, "u_bar2c" => -1.0, "u_bar2d" => 1.0,
                 "u_bar3a" => -1.0, "u_bar3b" => -1.0, "u_bar3c" =>  1.0, "u_bar3d" => 1.0,
                 "u_bar4a" => -1.0, "u_bar4b" => -1.0, "u_bar4c" =>  1.0)[name]
u̇_grid[1] = tangent(u_grid[1])
if u̇_grid[1][end] * direction < 0
    u̇_grid[1] *= -1
end

K_A = 165 # chosen independently from K
Π_K_A  = Projection(evensym(Fourier(K_A, π)) × evensym(Fourier(K_A, π)) × ScalarSpace())
Π_2K_A = Projection(evensym(Fourier(2K_A, π)) × evensym(Fourier(2K_A, π)) × ScalarSpace())
A_finite_grid[1] = approx_A(u_grid[1], u̇_grid[1], Π_K_A, Π_2K_A)

# run continuation scheme

for j ∈ 2:N+1
    δ = arclength_grid[j] - arclength_grid[j-1]

    predictor = u_grid[j-1] + δ * u̇_grid[j-1]

    u_barⱼ, convergedⱼ = newton(predictor) do u
        return F_pa(unpack(u), unpack(u̇_grid[j-1]), unpack(predictor)),
               DF_pa(unpack(u), unpack(u̇_grid[j-1]))
    end
    convergedⱼ || error("Newton failed")

    u_grid[j] = u_barⱼ

    u̇_grid[j] = tangent(u_grid[j])
    if sum(coefficients(u̇_grid[j-1]) .* coefficients(u̇_grid[j])) < 0 # keep the same direction
        u̇_grid[j] *= -1
    end

    A_finite_grid[j] = approx_A(u_grid[j], u̇_grid[j], Π_K_A, Π_2K_A)
end

# data[name] = collect(Float64, coefficients(u_grid[1]))
# name_next = Dict("u_bar1a" => "u_bar1b", "u_bar1b" => "u_bar1c",
#         "u_bar2a" => "u_bar2b", "u_bar2b" => "u_bar2c", "u_bar2c" => "u_bar2d",
#         "u_bar3a" => "u_bar3b", "u_bar3b" => "u_bar3c", "u_bar3c" => "u_bar3d",
#         "u_bar4a" => "u_bar4b", "u_bar4b" => "u_bar4c")
# if haskey(name_next, name)
#     data[name_next[name]] = collect(Float64, coefficients(u_grid[end]))
# end
# JLD2.save(joinpath(@__DIR__, "skt_data.jld2"), data)

#

function approx_inverse_DvΦ(u, K_A)
    ub = unpack(u)
    U = DvΦ(ub[1:2], ub[3][1])
    m = 2K_A + 1
    U_grid = [real.(to_grid(U[i,j], m)) for i ∈ 1:2, j ∈ 1:2]
    W_pts = [inv([U_grid[i,j][l] for i ∈ 1:2, j ∈ 1:2]) for l ∈ 1:m]
    return [real(to_coef([W[i,j] for W ∈ W_pts], evensym(Fourier(K_A, π)))) for i ∈ 1:2, j ∈ 1:2]
end

W_grid = [approx_inverse_DvΦ(u, K_A) for u ∈ u_grid];

# retrieve the Chebyshev interpolants

u_cheb = interval(real(to_coef(interval.(u_grid), Chebyshev(N))));

u̇_cheb = interval(real(to_coef(interval.(u̇_grid), Chebyshev(N))));

A_finite_cheb = interval(real(to_coef(A_finite_grid, Chebyshev(N))));

W_cheb = [interval(real(to_coef(getindex.(W_grid, i, j), Chebyshev(N)))) for i ∈ 1:2, j ∈ 1:2];





# Step 3: Estimating the bounds

# utilitary

struct PseudoInverseLaplacian <: AbstractDiagonalOperator end
RadiiPolynomial.getcoefficient(::PseudoInverseLaplacian, (codom, i)::Tuple{SymmetricSpace{<:Fourier},Integer}, (dom, j)::Tuple{SymmetricSpace{<:Fourier},Integer}) =
    (i == j) & !(i == j == 0) ? inv(-(frequency(dom) * exact(i))^2) : zero(frequency(dom))

Δ⁻¹ = PseudoInverseLaplacian()



#- Y bound

N_Y = 3N;

u_grid_Y = real.(to_grid(u_cheb, N_Y+1));

u̇_grid_Y = real.(to_grid(u̇_cheb, N_Y+1));

A_finite_grid_Y = real.(to_grid(A_finite_cheb, N_Y+1));

W_grid_Y_ = [real.(to_grid(W_cheb[i,j], N_Y+1)) for i ∈ 1:2, j ∈ 1:2];
W_grid_Y = [getindex.(W_grid_Y_, j) for j ∈ eachindex(W_grid_Y_[1])];

Πseq_K_A = interval(Projection(evensym(Fourier(K_A, π))));
zero_col = mapreduce(_ -> interval(LinearOperator(ScalarSpace(), evensym(Fourier(0, π)), [0;;])), vcat, 1:2);
zero_row = mapreduce(_ -> interval(LinearOperator(evensym(Fourier(0, π)), ScalarSpace(), [0;;])), hcat, 1:2);
zero_corner = interval(LinearOperator(ScalarSpace(), ScalarSpace(), [0;;]));
A_tail_grid_Y = map(W_grid_Y) do W_
    W = Multiplication.(W_)
    [W .* Δ⁻¹ .- Πseq_K_A .* (W .* Δ⁻¹) .* Πseq_K_A    zero_col
     zero_row                                          zero_corner]
end;

A_grid_Y = unpack.(A_finite_grid_Y) .+ A_tail_grid_Y;

#--

Π_2K = interval(Projection(evensym(Fourier(2K, π)) × evensym(Fourier(2K, π)) × ScalarSpace()));

Y_grid = A_grid_Y .* (Π_2K .* F_pa.(unpack.(u_grid_Y), unpack.(u̇_grid_Y), unpack.(u_grid_Y)));
Y = norm(real(to_coef(Y_grid, Chebyshev(N_Y))), 1)



#- Z₁ bound

N_Z = 2N;

u_grid_Z = real.(to_grid(u_cheb, N_Z+1));

u̇_grid_Z = real.(to_grid(u̇_cheb, N_Z+1));

A_finite_grid_Z = real.(to_grid(A_finite_cheb, N_Z+1));

W_grid_Z_ = [real.(to_grid(W_cheb[i,j], N_Z+1)) for i ∈ 1:2, j ∈ 1:2];
W_grid_Z = [getindex.(W_grid_Z_, j) for j ∈ eachindex(W_grid_Z_[1])];

A_tail_grid_Z = map(W_grid_Z) do W_
    W = Multiplication.(W_)
    [W .* Δ⁻¹ .- Πseq_K_A .* (W .* Δ⁻¹) .* Πseq_K_A    zero_col
     zero_row                                          zero_corner]
end;

A_grid_Z = unpack.(A_finite_grid_Z) .+ A_tail_grid_Z;

#--

Π_K_K_A = interval(Projection(evensym(Fourier(K+K_A, π)) × evensym(Fourier(K+K_A, π)) × ScalarSpace()));

Z₁_finite = opnorm(real(to_coef(Π_K_K_A .- A_grid_Z .* DF_pa.(unpack.(u_grid_Z), unpack.(u̇_grid_Z)) .* Π_K_K_A, Chebyshev(N_Z))), 1)

#--

M_grid = Ref(exact(I(2))) .- W_grid_Z .* DvΦ.(getindex.(unpack.(u_grid_Z), Ref(1:2)), getindex.(getindex.(unpack.(u_grid_Z), 3), 1));
M_cheb = [real(to_coef([W[i,j] for W ∈ M_grid], Chebyshev(N_Z))) for i ∈ 1:2, j ∈ 1:2];

DR_grid = DR.(getindex.(unpack.(u_grid_Z), Ref(1:2)));
DR_cheb = [real(to_coef([DR[i,j] for DR ∈ DR_grid], Chebyshev(N_Z))) for i ∈ 1:2, j ∈ 1:2];

Z₁_tail = opnorm(norm.(M_cheb, 1), 1) +
    opnorm(norm.(W_cheb, 1), 1) * opnorm(norm.(DR_cheb, 1), 1) / (interval(π) * interval(K_A + 1))^2

#--

Z₁ = max(Z₁_finite, Z₁_tail)



#- Z₂ bound

r_star = Inf

u_grid_rig = real.(to_grid(u_cheb, N+1));

u̇_grid_rig = real.(to_grid(u̇_cheb, N+1));

A_finite_grid_rig = real.(to_grid(A_finite_cheb, N+1));

W_grid_rig_ = [real.(to_grid(W_cheb[i,j], N+1)) for i ∈ 1:2, j ∈ 1:2];
W_grid_rig = [getindex.(W_grid_rig_, j) for j ∈ eachindex(W_grid_rig_[1])];

Δ = Laplacian();

opnormW = opnorm(norm.(W_cheb, 1), 1)

A_tail_grid_rig = map(W_grid_rig) do W_
    W = Multiplication.(W_)
    [W .* Δ⁻¹ .- Πseq_K_A .* (W .* Δ⁻¹) .* Πseq_K_A    zero_col
     zero_row                                          zero_corner]
end;

A_grid_rig = unpack.(A_finite_grid_rig) .+ A_tail_grid_rig;
A_cheb_trunc = real(to_coef(A_grid_rig .* interval(Π_K_A), Chebyshev(N)));
opnormA = max(opnorm(A_cheb_trunc, 1), opnormW / (interval(π) * interval(K_A + 1))^2)

AΔ_grid_rig = (unpack.(A_finite_grid_rig) .+ A_tail_grid_rig) .*
            Ref([[Δ exact(0)*I ; exact(0)*I Δ]    zero_col
                 zero_row                         zero_corner]);
AΔ_cheb_trunc = real(to_coef(AΔ_grid_rig .* interval(Π_K_A), Chebyshev(N)));
opnormAΔ = max(opnorm(A_cheb_trunc, 1), opnormW)

Z₂ = opnormAΔ * interval(max(2*d₁₁, d₁₂ + d₂₁, 2d₂₂, 1)) + opnormA * interval(max(2a₁, b₁ + b₂, 2a₂))



#-

ie, proved = interval_of_existence(Y, Z₁, Z₂, r_star; verbose = true)
proved || error("Proof failed")





# Plot

lines!(ax_skt, [Point2f(mid(real(component(u_cheb, 3)(s))), mid(real(component(u_cheb, 2)(s, 0)))) for s ∈ LinRange(-1, 1, 501)];
    color = ifelse(isodd(iter), :blue, :green), linewidth = 2)

scatter!(ax_skt, [Point2f(u_grid[j][end], real(component(u_grid[j], 2)(0))) for j ∈ 1:N+1];
    color = ifelse(isodd(iter), :blue, :green))

display(fig_skt)
end
