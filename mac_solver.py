import numpy as np
from numba import njit, prange

# ============================================================
# Utils
# ============================================================

@njit(inline="always")
def clamp(x, a, b):
    if x < a:
        return a
    if x > b:
        return b
    return x

# ============================================================
# Amostragem bilinear em grids MAC
# ============================================================

@njit(inline="always")
def sample_u(u, x, y, x_min, y_min, dx, dy, nx, ny):
    """
    u: (nx+3, ny+2) com ghosts
    u-face em x = x_min + i*dx, y = y_min + (j-0.5)*dy, i=1..nx+1, j=1..ny
    """
    fx = (x - x_min) / dx
    fy = (y - y_min) / dy + 0.5

    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0

    i0 = clamp(i0, 0, nx+1)
    j0 = clamp(j0, 0, ny)

    i1 = i0 + 1
    j1 = j0 + 1

    if i1 > nx+2: i1 = nx+2
    if j1 > ny+1: j1 = ny+1

    v00 = u[i0, j0]
    v10 = u[i1, j0]
    v01 = u[i0, j1]
    v11 = u[i1, j1]

    a = v00*(1.0 - tx) + v10*tx
    b = v01*(1.0 - tx) + v11*tx
    return a*(1.0 - ty) + b*ty

@njit(inline="always")
def sample_v(v, x, y, x_min, y_min, dx, dy, nx, ny):
    """
    v: (nx+2, ny+3) com ghosts
    v-face em x = x_min + (i-0.5)*dx, y = y_min + j*dy, i=1..nx, j=1..ny+1
    """
    fx = (x - x_min) / dx + 0.5
    fy = (y - y_min) / dy

    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0

    i0 = clamp(i0, 0, nx)
    j0 = clamp(j0, 0, ny+1)

    i1 = i0 + 1
    j1 = j0 + 1

    if i1 > nx+1: i1 = nx+1
    if j1 > ny+2: j1 = ny+2

    v00 = v[i0, j0]
    v10 = v[i1, j0]
    v01 = v[i0, j1]
    v11 = v[i1, j1]

    a = v00*(1.0 - tx) + v10*tx
    b = v01*(1.0 - tx) + v11*tx
    return a*(1.0 - ty) + b*ty

@njit(inline="always")
def vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny):
    return (
        sample_u(u, x, y, x_min, y_min, dx, dy, nx, ny),
        sample_v(v, x, y, x_min, y_min, dx, dy, nx, ny),
    )

# ============================================================
# BCs - Cavity (mantido)
# ============================================================

@njit
def apply_bc_cavity(u, v, p, nx, ny, U_lid):
    # Pressão Neumann
    for j in range(0, ny+2):
        p[0, j]    = p[1, j]
        p[nx+1, j] = p[nx, j]
    for i in range(0, nx+2):
        p[i, 0]    = p[i, 1]
        p[i, ny+1] = p[i, ny]
    p[1, 1] = 0.0

    # u paredes
    for j in range(1, ny+1):
        u[1, j]    = 0.0
        u[nx+1, j] = 0.0
        u[0, j]    = -u[1, j]
        u[nx+2, j] = -u[nx+1, j]

    for i in range(1, nx+2):
        u[i, 0]    = -u[i, 1]                 # bottom
        u[i, ny+1] = 2.0*U_lid - u[i, ny]     # top lid

    # v paredes
    for i in range(1, nx+1):
        v[i, 1]    = 0.0
        v[i, ny+1] = 0.0
        v[i, 0]    = -v[i, 1]
        v[i, ny+2] = -v[i, ny+1]

    for j in range(1, ny+2):
        v[0, j]    = -v[1, j]
        v[nx+1, j] = -v[nx, j]

# ============================================================
# BCs - Airfoil farfield/outlet (novo)
# ============================================================

@njit
def apply_bc_airfoil(u, v, p, nx, ny, Ux, Uy):
    """
    Domínio: esquerda=inlet Dirichlet, topo/baixo=farfield Dirichlet,
    direita=outlet Neumann (copia interior).
    Pressão: Neumann em tudo + p[1,1]=0 (remove modo nulo).
    """
    # -------- Pressure Neumann
    for j in range(0, ny+2):
        p[0, j]    = p[1, j]
        p[nx+1, j] = p[nx, j]
    for i in range(0, nx+2):
        p[i, 0]    = p[i, 1]
        p[i, ny+1] = p[i, ny]
    p[1, 1] = 0.0

    # -------- u Dirichlet no inlet/top/bottom
    # u faces: i=1..nx+1, j=1..ny
    for j in range(1, ny+1):
        # inlet (x_min): i=1
        u[1, j] = Ux
        u[0, j] = 2.0*Ux - u[1, j]

        # outlet (x_max): i=nx+1 -> grad=0
        u[nx+1, j] = u[nx, j]
        u[nx+2, j] = u[nx+1, j]

    for i in range(1, nx+2):
        # bottom/top farfield
        u[i, 0]    = 2.0*Ux - u[i, 1]
        u[i, ny+1] = 2.0*Ux - u[i, ny]

    # -------- v Dirichlet no inlet/top/bottom
    # v faces: i=1..nx, j=1..ny+1
    for j in range(1, ny+2):
        # inlet/outlet ghosts laterais (para v) - impondo Uy por simetria
        v[1, j] = v[1, j]  # interior já será arrastado; não força aqui direto

        # left ghost:
        v[0, j] = 2.0*Uy - v[1, j]
        # right ghost:
        v[nx+1, j] = v[nx, j]

    for i in range(1, nx+1):
        # bottom/top farfield: v = Uy
        v[i, 1]    = Uy
        v[i, 0]    = 2.0*Uy - v[i, 1]
        v[i, ny+1] = Uy
        v[i, ny+2] = 2.0*Uy - v[i, ny+1]

# ============================================================
# Advecção semi-Lagrangiana (RK2) - in-place
# ============================================================

@njit(parallel=True)
def advect_semi_lagrangian(u0, v0, u1, v1, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny):
    # advectar u
    for j in prange(1, ny+1):
        y = y_min + (j - 0.5) * dy
        for i in range(1, nx+2):
            x = x_min + i * dx

            uvel, vvel = vel_at(u0, v0, x, y, x_min, y_min, dx, dy, nx, ny)
            xm = x - 0.5 * dt * uvel
            ym = y - 0.5 * dt * vvel
            uvel2, vvel2 = vel_at(u0, v0, xm, ym, x_min, y_min, dx, dy, nx, ny)
            xb = x - dt * uvel2
            yb = y - dt * vvel2

            # clamp backtrace no domínio
            xb = clamp(xb, x_min, x_max)
            yb = clamp(yb, y_min, y_max)

            u1[i, j] = sample_u(u0, xb, yb, x_min, y_min, dx, dy, nx, ny)

    # advectar v
    for j in prange(1, ny+2):
        y = y_min + j * dy
        for i in range(1, nx+1):
            x = x_min + (i - 0.5) * dx

            uvel, vvel = vel_at(u0, v0, x, y, x_min, y_min, dx, dy, nx, ny)
            xm = x - 0.5 * dt * uvel
            ym = y - 0.5 * dt * vvel
            uvel2, vvel2 = vel_at(u0, v0, xm, ym, x_min, y_min, dx, dy, nx, ny)
            xb = x - dt * uvel2
            yb = y - dt * vvel2

            xb = clamp(xb, x_min, x_max)
            yb = clamp(yb, y_min, y_max)

            v1[i, j] = sample_v(v0, xb, yb, x_min, y_min, dx, dy, nx, ny)

# ============================================================
# Difusão explícita - in-place
# ============================================================

@njit(parallel=True)
def diffuse_inplace(u_in, v_in, u_out, v_out, nu, dt, dx, dy, nx, ny):
    idx2 = 1.0/(dx*dx)
    idy2 = 1.0/(dy*dy)

    for j in prange(1, ny+1):
        for i in range(1, nx+2):
            lap = (u_in[i+1,j]-2*u_in[i,j]+u_in[i-1,j])*idx2 + (u_in[i,j+1]-2*u_in[i,j]+u_in[i,j-1])*idy2
            u_out[i,j] = u_in[i,j] + nu*dt*lap

    for j in prange(1, ny+2):
        for i in range(1, nx+1):
            lap = (v_in[i+1,j]-2*v_in[i,j]+v_in[i-1,j])*idx2 + (v_in[i,j+1]-2*v_in[i,j]+v_in[i,j-1])*idy2
            v_out[i,j] = v_in[i,j] + nu*dt*lap

# ============================================================
# Brinkman penalization (IB) - in-place
# ============================================================

@njit(parallel=True)
def brinkman_penalize(u, v, chi_u, chi_v, dt, eta, nx, ny):
    # u: (nx+3, ny+2), chi_u: (nx+1, ny) -> mapeia u[1..nx+1,1..ny]
    for j in prange(1, ny+1):
        for i in range(1, nx+2):
            chi = chi_u[i-1, j-1]
            if chi > 0.0:
                u[i,j] = u[i,j] / (1.0 + dt*chi/eta)

    # v: (nx+2, ny+3), chi_v: (nx, ny+1) -> mapeia v[1..nx,1..ny+1]
    for j in prange(1, ny+2):
        for i in range(1, nx+1):
            chi = chi_v[i-1, j-1]
            if chi > 0.0:
                v[i,j] = v[i,j] / (1.0 + dt*chi/eta)

# ============================================================
# Divergência, Poisson (PCG), projeção
# ============================================================

@njit(parallel=True)
def compute_divergence(u, v, div, dx, dy, nx, ny):
    invdx = 1.0/dx
    invdy = 1.0/dy
    for j in prange(1, ny+1):
        for i in range(1, nx+1):
            div[i,j] = (u[i+1,j]-u[i,j])*invdx + (v[i,j+1]-v[i,j])*invdy

@njit(parallel=True)
def apply_A(p, Ap, dx, dy, nx, ny):
    idx2 = 1.0/(dx*dx)
    idy2 = 1.0/(dy*dy)
    for j in prange(1, ny+1):
        for i in range(1, nx+1):
            Ap[i,j] = (p[i+1,j]-2*p[i,j]+p[i-1,j])*idx2 + (p[i,j+1]-2*p[i,j]+p[i,j-1])*idy2

@njit
def dot_inner(a, b, nx, ny):
    s = 0.0
    for j in range(1, ny+1):
        for i in range(1, nx+1):
            s += a[i,j]*b[i,j]
    return s

@njit
def pcg_poisson_inplace(p, b, r, z, d, Ap, dx, dy, nx, ny, max_iter=200, tol=1e-6):
    apply_A(p, Ap, dx, dy, nx, ny)
    for j in range(1, ny+1):
        for i in range(1, nx+1):
            r[i,j] = b[i,j] - Ap[i,j]

    diag = -2.0*(1.0/(dx*dx) + 1.0/(dy*dy))
    invdiag = 1.0/diag

    for j in range(1, ny+1):
        for i in range(1, nx+1):
            z[i,j] = r[i,j] * invdiag
            d[i,j] = z[i,j]

    rz_old = dot_inner(r, z, nx, ny)
    bnorm = np.sqrt(max(dot_inner(b, b, nx, ny), 1e-30))

    for it in range(max_iter):
        apply_A(d, Ap, dx, dy, nx, ny)
        denom = dot_inner(d, Ap, nx, ny)
        if abs(denom) < 1e-30:
            return it + 1

        alpha = rz_old / denom

        for j in range(1, ny+1):
            for i in range(1, nx+1):
                p[i,j] += alpha * d[i,j]
                r[i,j] -= alpha * Ap[i,j]

        rnorm = np.sqrt(dot_inner(r, r, nx, ny)) / bnorm
        if rnorm < tol:
            return it + 1

        for j in range(1, ny+1):
            for i in range(1, nx+1):
                z[i,j] = r[i,j] * invdiag

        rz_new = dot_inner(r, z, nx, ny)
        beta = rz_new / rz_old
        rz_old = rz_new

        for j in range(1, ny+1):
            for i in range(1, nx+1):
                d[i,j] = z[i,j] + beta * d[i,j]

    return max_iter

@njit(parallel=True)
def project(u, v, p, dt, rho, dx, dy, nx, ny):
    invdx = 1.0/dx
    invdy = 1.0/dy
    scale = dt/rho

    for j in prange(1, ny+1):
        for i in range(1, nx+2):
            u[i,j] -= scale * (p[i,j]-p[i-1,j]) * invdx

    for j in prange(1, ny+2):
        for i in range(1, nx+1):
            v[i,j] -= scale * (p[i,j]-p[i,j-1]) * invdy

# ============================================================
# Força aerodinâmica via penalização (cell-centered)
# ============================================================

def compute_force_coeffs(u, v, chi_cell, rho, Uinf, alpha_rad, dx, dy):
    # cell-centered velocity
    nx, ny = chi_cell.shape
    uc = np.zeros((nx, ny), dtype=np.float32)
    vc = np.zeros((nx, ny), dtype=np.float32)

    # u array tem i=1..nx+1 faces; cell center i -> média u[i] e u[i+1]
    for j in range(1, ny+1):
        for i in range(1, nx+1):
            uc[i-1, j-1] = 0.5*(u[i, j] + u[i+1, j])
            vc[i-1, j-1] = 0.5*(v[i, j] + v[i, j+1])

    # força aproximada: -rho * (chi/eta)*u * dA (eta entra fora: usamos eta=1 aqui e reescala no chamador se quiser)
    # como estamos usando atualização u/(1+dt*chi/eta), a força coerente vem do termo (chi/eta)*u.
    # aqui retornamos apenas componentes proporcionais a sum(chi*u).
    Fx = -rho * float(np.sum(chi_cell * uc) * dx * dy)
    Fy = -rho * float(np.sum(chi_cell * vc) * dx * dy)

    ca = np.cos(alpha_rad); sa = np.sin(alpha_rad)
    D = Fx*ca + Fy*sa
    L = -Fx*sa + Fy*ca

    q = 0.5 * rho * Uinf * Uinf
    Sref = 1.0 * 1.0  # corda=1, espessura de 1 (2D por unidade de envergadura)
    Cd = D / (q * Sref + 1e-30)
    Cl = L / (q * Sref + 1e-30)

    return Cl, Cd, uc, vc

# ============================================================
# Simulações
# ============================================================

def simulate_cavity(nx, ny, steps, dt, rho, nu, U_lid, out_every=200):
    dx = 1.0 / nx
    dy = 1.0 / ny

    u = np.zeros((nx+3, ny+2), dtype=np.float32)
    v = np.zeros((nx+2, ny+3), dtype=np.float32)
    p = np.zeros((nx+2, ny+2), dtype=np.float32)

    u_adv = np.zeros_like(u)
    v_adv = np.zeros_like(v)
    u_star = np.zeros_like(u)
    v_star = np.zeros_like(v)

    div = np.zeros_like(p)
    b = np.zeros_like(p)

    r = np.zeros_like(p)
    z = np.zeros_like(p)
    d = np.zeros_like(p)
    Ap = np.zeros_like(p)

    apply_bc_cavity(u, v, p, nx, ny, U_lid)

    # domínio 0..1
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    for n in range(1, steps+1):
        advect_semi_lagrangian(u, v, u_adv, v_adv, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny)
        diffuse_inplace(u_adv, v_adv, u_star, v_star, nu, dt, dx, dy, nx, ny)
        apply_bc_cavity(u_star, v_star, p, nx, ny, U_lid)

        compute_divergence(u_star, v_star, div, dx, dy, nx, ny)
        scale = rho/dt
        b[1:nx+1, 1:ny+1] = scale * div[1:nx+1, 1:ny+1]

        apply_bc_cavity(u_star, v_star, p, nx, ny, U_lid)
        iters = pcg_poisson_inplace(p, b, r, z, d, Ap, dx, dy, nx, ny, max_iter=200, tol=1e-6)
        apply_bc_cavity(u_star, v_star, p, nx, ny, U_lid)

        project(u_star, v_star, p, dt, rho, dx, dy, nx, ny)
        apply_bc_cavity(u_star, v_star, p, nx, ny, U_lid)

        u, u_star = u_star, u
        v, v_star = v_star, v

        if out_every and (n % out_every == 0):
            dnorm = float(np.sqrt(np.mean(div[1:nx+1,1:ny+1]**2)))
            print(f"[cavity] step {n}/{steps} | pcg {iters} | div_rms {dnorm:.3e}")

    uc = np.zeros((nx, ny), dtype=np.float32)
    vc = np.zeros((nx, ny), dtype=np.float32)
    for j in range(1, ny+1):
        for i in range(1, nx+1):
            uc[i-1,j-1] = 0.5*(u[i,j] + u[i+1,j])
            vc[i-1,j-1] = 0.5*(v[i,j] + v[i,j+1])

    return {"p": p[1:nx+1,1:ny+1].copy(), "uc": uc, "vc": vc}

def simulate_airfoil_naca4412(
    nx, ny, steps, dt,
    rho, nu,
    Uinf, alpha_deg,
    x_min, x_max, y_min, y_max,
    chi_cell, eta=1e-3,
    out_every=200
):
    """
    chi_cell: (nx,ny) 1 dentro do aerofólio.
    """
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    u = np.zeros((nx+3, ny+2), dtype=np.float32)
    v = np.zeros((nx+2, ny+3), dtype=np.float32)
    p = np.zeros((nx+2, ny+2), dtype=np.float32)

    u_adv = np.zeros_like(u)
    v_adv = np.zeros_like(v)
    u_star = np.zeros_like(u)
    v_star = np.zeros_like(v)

    div = np.zeros_like(p)
    b = np.zeros_like(p)

    r = np.zeros_like(p)
    z = np.zeros_like(p)
    d = np.zeros_like(p)
    Ap = np.zeros_like(p)

    alpha = np.deg2rad(alpha_deg)
    Ux = Uinf * np.cos(alpha)
    Uy = Uinf * np.sin(alpha)

    # Construir chi_u e chi_v a partir de chi_cell (rápido e suficiente)
    # chi_u (nx+1, ny): faces verticais entre células
    chi_u = np.zeros((nx+1, ny), dtype=np.float32)
    for j in range(ny):
        for i in range(nx+1):
            left = chi_cell[i-1, j] if i-1 >= 0 else 0.0
            right = chi_cell[i, j] if i < nx else 0.0
            chi_u[i, j] = 1.0 if (left > 0.0 or right > 0.0) else 0.0

    # chi_v (nx, ny+1): faces horizontais entre células
    chi_v = np.zeros((nx, ny+1), dtype=np.float32)
    for j in range(ny+1):
        for i in range(nx):
            bot = chi_cell[i, j-1] if j-1 >= 0 else 0.0
            top = chi_cell[i, j] if j < ny else 0.0
            chi_v[i, j] = 1.0 if (bot > 0.0 or top > 0.0) else 0.0

    # Inicializa com freestream
    u[1:nx+2, 1:ny+1] = Ux
    v[1:nx+1, 1:ny+2] = Uy
    apply_bc_airfoil(u, v, p, nx, ny, Ux, Uy)

    Cl = Cd = 0.0
    for n in range(1, steps+1):
        # advecção
        advect_semi_lagrangian(u, v, u_adv, v_adv, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny)

        # difusão
        diffuse_inplace(u_adv, v_adv, u_star, v_star, nu, dt, dx, dy, nx, ny)

        # penalização do sólido
        brinkman_penalize(u_star, v_star, chi_u, chi_v, dt, eta, nx, ny)

        # BCs
        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)

        # Poisson
        compute_divergence(u_star, v_star, div, dx, dy, nx, ny)
        scale = rho/dt
        b[1:nx+1, 1:ny+1] = scale * div[1:nx+1, 1:ny+1]

        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)
        iters = pcg_poisson_inplace(p, b, r, z, d, Ap, dx, dy, nx, ny, max_iter=250, tol=1e-6)
        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)

        # projeção
        project(u_star, v_star, p, dt, rho, dx, dy, nx, ny)
        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)

        # swap
        u, u_star = u_star, u
        v, v_star = v_star, v

        if out_every and (n % out_every == 0):
            dnorm = float(np.sqrt(np.mean(div[1:nx+1,1:ny+1]**2)))
            # coeficientes (aprox rápido)
            Cl, Cd, _, _ = compute_force_coeffs(u, v, chi_cell, rho, Uinf, alpha, dx, dy)
            print(f"[airfoil] step {n}/{steps} | pcg {iters} | div_rms {dnorm:.3e} | Cl~{Cl:.3f} Cd~{Cd:.3f}")

    Cl, Cd, uc, vc = compute_force_coeffs(u, v, chi_cell, rho, Uinf, alpha, dx, dy)
    return {
        "p": p[1:nx+1, 1:ny+1].copy(),
        "uc": uc,
        "vc": vc,
        "Cl": Cl,
        "Cd": Cd,
        "dx": dx,
        "dy": dy,
    }
