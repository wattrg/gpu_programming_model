using CUDA
using BenchmarkTools

struct FlowStates
    p
    T
    rho
    vx
    vy
    vz
end

mutable struct Conserved
    mass
    px
    py
    pz
    e
end

function ausmdv_arrays!(left::FlowStates, right::FlowStates, flux::Conserved)
    rL = left.rho
    pL = left.p
    pLrL = pL ./ rL
    uL = left.vx
    vL = left.vy
    wL = left.vz
    eL = Cv .* left.T
    aL = CUDA.sqrt.(gamma .* R .* left.T)
    keL = @. 0.5 * (uL * uL + vL * vL + wL * wL)
    HL = @. eL + pLrL + keL

    rR = right.rho
    pR = right.p
    pRrR = pR ./ rR
    uR = right.vx
    vR = right.vy
    wR = right.vz
    eR = Cv .* right.T
    aR = CUDA.sqrt.(gamma .* R .* right.T)
    keR = @. 0.5 * (uR * uR + vR * vR + wR * wR)
    HR = @. eR + pRrR + keR

    alphaL = @. 2.0 * pLrL / (pLrL + pRrR)
    alphaR = @. 2.0 * pRrR / (pLrL + pRrR)

    am = CUDA.max.(aL, aR)
    ML = uL ./ am
    MR = uR ./ am

    duL = @. 0.5 * (uL + CUDA.abs(uL))
    pLplus = similar(pL)
    uLplus = similar(pL)
    subsonic = @. CUDA.abs(ML) <= 1.0
    subsonic_pLplus = @. pL * (ML + 1.0) * (ML + 1.0) * (2.0 - ML) * 0.25
    subsonic_uLplus = @. alphaL * ((uL +am) * (uL + am) / (4.0 * am) - duL) + duL
    pLplus_supersonic = @. pL * duL / uL
    pLplus[subsonic] = subsonic_pLplus[subsonic]
    uLplus[subsonic] = subsonic_uLplus[subsonic]
    pLplus[.!subsonic] = pLplus_supersonic[.!subsonic]
    uLplus[.!subsonic] = duL[.!subsonic]

    duR = @. 0.5 * (uR - CUDA.abs(uR))
    pRminus = similar(pR)
    uRminus = similar(uR)
    subsonic = @. CUDA.abs(MR) <= 1.0
    pRminus_subsonic = @. pR * (MR - 1.0) * (MR - 1.0) * (2.0 + MR) * 0.25
    uRminus_subsonic = @. alphaR * (-(uR - am) * (uR - am) / (4.0 * am) - duR) + duR
    pRminus_supersonic = @. pR * duR / uR
    pRminus[subsonic] = pRminus_subsonic[subsonic]
    uRminus[subsonic] = uRminus_subsonic[subsonic]
    pRminus[.!subsonic] = pRminus_supersonic[.!subsonic]
    uRminus[.!subsonic] = duR[.!subsonic]

    ru_half = @. uLplus * rL + uRminus * rR
    p_half = @. pLplus + pRminus

    dp = @. pL - pR
    K_SWITCH = 10.0
    dp = @. K_SWITCH * CUDA.abs(dp) / CUDA.min(pL, pR)
    s = @. 0.5 * CUDA.min(1.0, dp)
    ru2_ausmv = @. uLplus * rL * uL + uRminus * rR * uR
    ru2_ausmd = @. 0.5 * (ru_half * (uL + uR) - CUDA.abs(ru_half) * (uR - uL)) 
    ru2_half = @. (0.5 + s) * ru2_ausmv + (0.5 - s) * ru2_ausmd 

    blowing_left = @. ru_half >= 0.0
    momentum_x_left = @. ru2_half + p_half
    momentum_y_left = @. ru_half * vL
    momentum_z_left = @. ru_half * wL
    energy_left = @. ru_half * HL
    momentum_x_right = @. ru2_half + p_half
    momentum_y_right = @. ru_half * vR
    momentum_z_right = @. ru_half * wR
    energy_right = @. ru_half * HR

    flux.mass = ru_half
    flux.px[blowing_left] = momentum_x_left[blowing_left]
    flux.px[.!blowing_left] = momentum_x_right[.!blowing_left]
    flux.py[blowing_left] = momentum_y_left[blowing_left]
    flux.py[.!blowing_left] = momentum_y_right[.!blowing_left]
    flux.pz[blowing_left] = momentum_z_left[blowing_left]
    flux.pz[.!blowing_left] = momentum_z_right[.!blowing_left]
    flux.e[blowing_left] = energy_left[blowing_left]
    flux.e[.!blowing_left] = energy_right[.!blowing_left]
end

N = 500000
R = 287.0
Cv = 5.0 / 2.0 * R
Cp = 7.0 / 2.0 * R
gamma = Cp / Cv

pL = CuArray(fill(101325.0, N))
TL = CuArray(fill(300.0, N))
rhoL = @. pL / (R * TL)
vxL = CuArray(LinRange(-2000, 2000, N))
vyL = CuArray(LinRange(-2000, 2000, N))
vzL = CuArray(LinRange(-2000, 2000, N))
left = FlowStates(pL, TL, rhoL, vxL, vyL, vzL)

pR = CuArray(fill(101325.0, N))
TR = CuArray(fill(300.0, N))
rhoR = @. pR / (R * TR)
vxR = CuArray(LinRange(-2000, 2000, N))
vyR = CuArray(LinRange(-2000, 2000, N))
vzR = CuArray(LinRange(-2000, 2000, N))
right = FlowStates(pR, TR, rhoR, vxR, vyR, vzR)

mass_flux = CuArray(zeros(N))
px_flux = CuArray(zeros(N))
py_flux = CuArray(zeros(N))
pz_flux = CuArray(zeros(N))
e_flux = CuArray(zeros(N))
flux = Conserved(mass_flux, px_flux, py_flux, pz_flux, e_flux)

@btime ausmdv_arrays!(left, right, flux)
