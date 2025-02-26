using Plots
using ImplicitPlots
using Plots.PlotMeasures
using OptimalControl
using NLPModelsIpopt
using LaTeXStrings
using NonlinearSolve
using OrdinaryDiffEq
# using LinearAlgebra
include("smooth.jl")

# gr()

tf = 0.5

A1 = [-13. 18.; 12. -20.]
rho1 = [-11.0 / 20, 7.0 / 10]
c1 = -A1 * rho1

A2 = [-6. 1.; -2. -1.]
rho2 = [-0.8, -0.5]
c2 = -A2 * rho2

r1 = -A1 * [1, 1]
r2 = -A2 * [1, 1]

yx0 = 0.1
yy0 = 0.5
zx0 = 0.9
zy0 = 0.7
yz0 = [yx0, yy0, zx0, zy0]

rf = sqrt(5.58)
af = 3.5
bf = 1.5

r2f = 0.1
a2f = zx0 - 0.3
b2f = zy0

function eqN2(x, y)
    return (x - af)^2 + (y - bf)^2 - rf^2
end

function eqZ2(x, y)
    return (x - a2f)^2 + (y - b2f)^2 - r2f^2
end

G(x, t) = fNC_unboundedminus(x, t, 0.005)

ocp = @def begin
    t ∈ [0, tf], time
    q = [x1x, x1y, x2x, x2y, z1x, z1y, z2x, z2y] ∈ R^8, state
    u = [u1, u2] ∈ R^2, control
    ps = [tau, cc] ∈ R^2, variable

    x1x(0) == yx0
    x1y(0) == yy0
    z1x(0) == zx0
    z1y(0) == zy0
    x2x(0) - x1x(tf) - cc * z1x(tf) == 0
    x2y(0) - x1y(tf) - cc * z1y(tf) == 0
    z2x(0) - z1x(tf) == 0
    z2y(0) - z1y(tf) == 0
    0 <= tau <= tf
    0 <= cc <= 1
    (x2x(tf) - af)^2 + (x2y(tf) - bf)^2 <= rf^2
    (z2x(tf) - a2f)^2 + (z2y(tf) - b2f)^2 <= r2f^2

    0 <= u1(t) <= 1
    0 <= u2(t) <= 1

    derivative(q)(t) == [
        x1x(t) * (A1[1] * x1x(t) + A1[3] * x1y(t) + r1[1] + u1(t) * c1[1]) * G(t, tau),
        x1y(t) * (A1[2] * x1x(t) + A1[4] * x1y(t) + r1[2] + u1(t) * c1[2]) * G(t, tau),
        x2x(t) * (A1[1] * x2x(t) + A1[3] * x2y(t) + r1[1] + u1(t) * c1[1]) * (1 - G(t, tau)),
        x2y(t) * (A1[2] * x2x(t) + A1[4] * x2y(t) + r1[2] + u1(t) * c1[2]) * (1 - G(t, tau)),
        z1x(t) * (A2[1] * z1x(t) + A2[3] * z1y(t) + r2[1] + u2(t) * c2[1]) * G(t, tau),
        z1y(t) * (A2[2] * z1x(t) + A2[4] * z1y(t) + r2[2] + u2(t) * c2[2]) * G(t, tau),
        z2x(t) * (A2[1] * z2x(t) + A2[3] * z2y(t) + r2[1] + u2(t) * c2[1]) * (1 - G(t, tau)),
        z2y(t) * (A2[2] * z2x(t) + A2[4] * z2y(t) + r2[2] + u2(t) * c2[2]) * (1 - G(t, tau))
    ]

    (x1x(tf) - x2x(0))^2 + (x1y(tf) - x2y(0))^2 → min
end

N = 500
sol = solve(ocp, max_iter=4000)
plot(sol)
savefig("all.pdf")

taud = sol.variable[1]
ccd = sol.variable[2]
ttd = sol.time_grid
tspan1 = ttd[ttd .< taud]
tspan2 = ttd[ttd .>= taud]

pxx0 = sol.costate(0)[1]
pxy0 = sol.costate(0)[2]
pzx0 = sol.costate(0)[3]
pzy0 = sol.costate(0)[4]

xxs = vcat([sol.state(t)[1] for t in tspan1], [sol.state(t)[3] for t in tspan2])
xys = vcat([sol.state(t)[2] for t in tspan1], [sol.state(t)[4] for t in tspan2])
zxs = vcat([sol.state(t)[5] for t in tspan1], [sol.state(t)[7] for t in tspan2])
zys = vcat([sol.state(t)[6] for t in tspan1], [sol.state(t)[8] for t in tspan2])

u1 = [sol.control(t)[1] for t in ttd]
u2 = [sol.control(t)[2] for t in ttd]

xxs_plot = plot(ttd, xxs, xlabel=L"t", ylabel=L"y_1", legend=false)
vline!([taud], label="", color=:red, linestyle=:dash)
xys_plot = plot(ttd, xys, xlabel=L"t", ylabel=L"y_2", legend=false)
vline!([taud], label="", color=:red, linestyle=:dash)
zxs_plot = plot(ttd, zxs, xlabel=L"t", ylabel=L"z_1", legend=false)
vline!([taud], label="", color=:red, linestyle=:dash)
zys_plot = plot(ttd, zys, xlabel=L"t", ylabel=L"z_2", legend=false)
vline!([taud], label="", color=:red, linestyle=:dash)
u1_plot = plot(ttd, u1, xlabel=L"t", ylabel=L"u_1", legend=false)
vline!([taud], label="", color=:red, linestyle=:dash)
u2_plot = plot(ttd, u2, xlabel=L"t", ylabel=L"u_2", legend=false)
vline!([taud], label="", color=:red, linestyle=:dash)

display(plot(xxs_plot, xys_plot, zxs_plot, zys_plot, u1_plot, u2_plot,
             layout=(3, 2), left_margin=10mm))
savefig("direct_method.pdf")

println(taud, " ", ccd)
xy_plot = plot(xxs, xys, legend=false, color=:blue, linestyle=:dot)
xy_plot = plot!(zxs, zys, legend=false, color=:green, linestyle=:dot)
xy_plot = implicit_plot!(eqN2; xlims=(0, 2), ylims=(0.45, 3), color=:red)
xy_plot = implicit_plot!(eqZ2; xlims=(0, 2), ylims=(0.45, 3), color=:red)
annotate!(-1.2, 1.5, text("Direct method:\n" * L"\bar\tau=%$(round(taud, digits=2))" *
                          "\n" * L"\bar\rho=%$(round(ccd[end], digits=2))", 8, :blue))

function F0(x)
    return [
        x[1] * (r1[1] + A1[1] * x[1] + A1[3] * x[2]),
        x[2] * (r1[2] + A1[2] * x[1] + A1[4] * x[2]),
        x[3] * (r2[1] + A2[1] * x[3] + A2[3] * x[4]),
        x[4] * (r2[2] + A2[2] * x[3] + A2[4] * x[4])
    ]
end

function F1(x)
    return [
        x[1] * c1[1],
        x[2] * c1[2],
        0.0,
        0.0
    ]
end

function F2(x)
    return [
        0.0,
        0.0,
        x[3] * c2[1],
        x[4] * c2[2]
    ]
end

H0(x, p) = p' * F0(x)
H1(x, p) = p' * F1(x)
H2(x, p) = p' * F2(x)
Hc(x, p, u1, u2) = H0(x, p) + u1 * H1(x, p) + u2 * H2(x, p)

up(x, p) = 1
um(x, p) = 0

Hpp(x, p) = Hc(x, p, up(x, p), up(x, p))
Hmp(x, p) = Hc(x, p, um(x, p), up(x, p))
Hpm(x, p) = Hc(x, p, up(x, p), um(x, p))
Hmm(x, p) = Hc(x, p, um(x, p), um(x, p))

fpp = Flow(Hamiltonian(Hpp))
fmm = Flow(Hamiltonian(Hmm))
fmp = Flow(Hamiltonian(Hmp))
fpm = Flow(Hamiltonian(Hpm))

function shoot(p0, t1, tau, c0, lbd)
    xzt1, pt1 = fmm(0.0, yz0, p0, t1)
    xzt2, pt2 = fpm(t1, xzt1, pt1, tau)
    x2plus = xzt2 + c0 * [xzt2[3:4]; 0; 0]
    p2plus = pt2
    p2plus[3:4] = pt2[3:4] + c0 * (2 * c0 * lbd * xzt2[3:4] - pt2[1:2])
    xf, pf = fmp(tau, x2plus, p2plus, tf)

    s = zeros(eltype(p0), 9)
    s[1] = (xf[1] - af)^2 + (xf[2] - bf)^2 - rf^2
    gdN = [2 * (xf[1] - af), 2 * (xf[2] - bf)]
    s[2:3] = pf[1:2] + gdN

    s[4] = (xf[3] - a2f)^2 + (xf[4] - b2f)^2 - r2f^2
    gdZ = [2 * (xf[3] - a2f), 2 * (xf[4] - b2f)]
    s[5:6] = pf[3:4] + gdZ

    s[7] = (2.0 * c0 * lbd * xzt2[3] - pt2[1]) * xzt2[3] +
           (2.0 * c0 * lbd * xzt2[4] - pt2[2]) * xzt2[4]
    s[8] = H1(xzt1, pt1)

    return s
end

nle! = (ξ, λ) -> shoot(ξ[1:4], ξ[5], ξ[6], ξ[7], ξ[8])

t1 = 0.18
t3 = 0.39
lbd = 0.5
xi_guess = [pxx0, pxy0, pzx0, pzy0, t1, taud, ccd, lbd]
xi_guess = [3.4157201368226957, 0.6320221044833143, 0.09567504890842388,
            0.7008002736539267, 0.21717961053867477, 0.40827568296737426,
            0.6688986083595068, 0.6998884087278446]

prob = NonlinearProblem(nle!, xi_guess)
indirect_sol = solve(prob; abstol=1e-8, reltol=1e-8, show_trace=Val(true))

p0 = indirect_sol[1:4]
t1 = indirect_sol[5]
tau0 = indirect_sol[6]
cc = indirect_sol[7]
lbd = indirect_sol[8]

ode_sol = fmm((0.0, t1), yz0, p0, saveat=0.01)
tt1 = ode_sol.t
xx1 = [ode_sol[1:4, j] for j in 1:size(tt1, 1)]
pp1 = [ode_sol[5:8, j] for j in 1:size(tt1, 1)]

ode_sol = fpm((t1, tau0), xx1[end], pp1[end], saveat=0.01)
tt3 = ode_sol.t
xx3 = [ode_sol[1:4, j] for j in 1:size(tt3, 1)]
pp3 = [ode_sol[5:8, j] for j in 1:size(tt3, 1)]

ode_sol = fmp((tau0, tf), xx3[end] + cc * [xx3[end][3:4]; 0.0; 0.0],
              pp3[end] + cc * [0; 0; 2.0 * cc * lbd * xx3[end][3:4] - pp3[end][1:2]],
              saveat=0.01)
tt4 = ode_sol.t
xx4 = [ode_sol[1:4, j] for j in 1:size(tt4, 1)]
pp4 = [ode_sol[5:8, j] for j in 1:size(tt4, 1)]

tt = [tt1; tt3; tt4]
x = [xx1; xx3; xx4]
p = [pp1; pp3; pp4]
u11 = [zeros(size(tt1, 1)); ones(size(tt3, 1)); zeros(size(tt4, 1))]
u22 = [zeros(size(tt1, 1)); zeros(size(tt3, 1)); ones(size(tt4, 1))]

m = length(tt)
x1 = [x[i][1] for i = 1:m]
x2 = [x[i][2] for i = 1:m]
x3 = [x[i][3] for i = 1:m]
x4 = [x[i][4] for i = 1:m]
p1 = [p[i][1] for i = 1:m]
p2 = [p[i][2] for i = 1:m]
p3 = [p[i][3] for i = 1:m]
p4 = [p[i][4] for i = 1:m]

q = [[x1[i], x2[i], x3[i], x4[i]] for i in 1:m]
p = [[p1[i], p2[i], p3[i], p4[i]] for i in 1:m]

xxs_plot = plot(ttd, xxs, label="direct method", legend=true, color=:blue,
                linestyle=:dot)
x1_plot = plot!(tt, x1, xlabel=L"t", ylabel=L"y_1", legend=true, color=:green,
                label="indirect method")
vline!([tau0], label=L"t=\bar\tau", color=:red, linestyle=:dash, legend=true)
xys_plot = plot(ttd, xys, xlabel=L"t", ylabel=L"y_2", legend=true, color=:blue,
                linestyle=:dot)
x2_plot = plot!(tt, x2, legend=false, color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
zxs_plot = plot(ttd, zxs, xlabel=L"t", ylabel=L"z_1", legend=false, color=:blue,
                linestyle=:dot)
x3_plot = plot!(tt, x3, legend=false, color=:green)
vline!([tau0], color=:red, linestyle=:dash)
zys_plot = plot(ttd, zys, xlabel=L"t", ylabel=L"z_2", legend=false, color=:blue,
                linestyle=:dot)
x4_plot = plot!(tt, x4, legend=false, color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
p1_plot = plot(tt, p1, xlabel=L"t", ylabel=L"{p_y}_1", label="indirect method",
               legend=true, color=:green)
vline!([tau0], label=L"t=\bar\tau", color=:red, linestyle=:dash, legend=true)
p2_plot = plot(tt, p2, xlabel=L"t", ylabel=L"{p_y}_2", legend=false, color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
p3_plot = plot(tt, p3, xlabel=L"t", ylabel=L"{p_z}_1", legend=false, color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
p4_plot = plot(tt, p4, xlabel=L"t", ylabel=L"{p_z}_2", legend=false, color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
u1_plot = plot(ttd, u1, xlabel=L"t", ylabel=L"u_1", size=(800, 400), color=:blue,
               linestyle=:dot)
u1_plot = plot!(tt, u11, legend=false, size=(800, 400), color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
u2_plot = plot(ttd, u2, xlabel=L"t", ylabel=L"u_2", legend=false, size=(800, 400),
               color=:blue, linestyle=:dot)
u2_plot = plot!(tt, u22, xlabel="t", ylabel=L"u_2", legend=false, size=(800, 400),
                color=:green)
vline!([tau0], label="", color=:red, linestyle=:dash)
xy_plot = plot(xxs, xys, legend=true, label=L"(y_1,y_2)\ (direct)", color=:blue)
xy_plot = plot!(zxs, zys, legend=true, label=L"(z_1,z_2)\ (direct)", color=:blue,
                linestyle=:dot)
xy_plot = plot!(x1, x2, legend=true, label=L"(y_1,y_2)\ (indirect)", color=:green,
                linestyle=:dashdot)
xy_plot = plot!(x3, x4, legend=:outertopleft, label=L"(z_1,z_2)\ (indirect)",
                color=:green, linestyle=:dashdotdot)
xy_plot = implicit_plot!(eqN2; xlims=(0, 1.7), ylims=(0.45, 2), legend=false,
                          color=:red)
xy_plot = implicit_plot!(eqZ2; xlims=(0, 1.7), ylims=(0.45, 2), legend=false,
                          color=:red)
annotate!(-1.2, 0.6, text("Indirect method:\n" * L"\bar\tau=%$(round(tau0, digits=2))" *
                          "\n" * L"\bar\rho=%$(round(cc, digits=2))" * "\n" *
                          L"\lambda=%$(round(lbd, digits=2))", 10, :green))
annotate!(-1.2, 1.5, text("Direct method:\n" * L"\bar\tau=%$(round(taud, digits=2))" *
                          "\n" * L"\bar\rho=%$(round(ccd[end], digits=2))", 10, :blue))
annotate!(0.5, 2.1, text(L"(y_1(t),y_2(t))", 10))
annotate!(0.7, 0.9, text(L"(z_1(t),z_2(t))", 10))
annotate!(1.25, 1.5, text(L"C_y", 9, :blue))
annotate!(0.78, 0.6, text(L"C_z", 9, :blue))

normz = sqrt.(x3 .^ 2 .+ x4 .^ 2)
println(xx3[end][3:4])
ztau = sqrt(sum(xx3[end][3:4] .^ 2))
println(ztau)

Hs_plot = plot(
    tt,
    sum((2 * lbd * cc * [x3 x4] * ztau - [p1 .* normz p2 .* normz]) .* [x3 x4],
        dims=2),
    xlabel=L"t",
    legend=false,
    ylabel=L"(2\bar\rho\lambda|\!\bar z(\bar\tau)|\!\bar z(t)-p_y(t)|\bar z(t)|)\cdot \bar z(t)",
    color=:green,
    yguidefontsize=10
)
vline!([tau0], label="", color=:red, linestyle=:dash)

plot1 = plot(p1_plot, p2_plot, p3_plot, p4_plot, xy_plot, Hs_plot, layout=(3, 2),
             size=(800, 600), left_margin=20mm)
display(plot1)
savefig("ps-multiprocessus.pdf")

plot2 = plot(x1_plot, x2_plot, x3_plot, x4_plot, u1_plot, u2_plot, layout=(3, 2),
             size=(800, 600), left_margin=10mm)
display(plot2)
savefig("xs-multiprocessus.pdf")

println(taud)
println(tau0)
println(ccd^2)
println(cc^2)

