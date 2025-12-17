# mac_solver.py
# MAC (staggered grid) incompressible Navier–Stokes 2D
# - Semi-Lagrangian RK2 advection (stable)
# - Explicit diffusion
# - Brinkman penalization (immersed body via chi)
# - Pressure projection via:
#     * Multigrid V-cycle (FAST, default)
#     * PCG Poisson (fallback)
# - Warm start (init_state)
# - Early stop (convergence of Cl and div_rms)
# - Force models:
#     * Brinkman
#     * Control Volume (momentum balance on a rectangular CV)  [FIXED dt_eff]
# - Optional time recording:
#     * Vorticity frames
#     * Passive tracers (particles)

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
# Bilinear sampling on MAC grids
# ============================================================

@njit(inline="always")
def sample_u(u, x, y, x_min, y_min, dx, dy, nx, ny):
    fx = (x - x_min) / dx
    fy = (y - y_min) / dy + 0.5

    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0

    i0 = clamp(i0, 0, nx + 1)
    j0 = clamp(j0, 0, ny)

    i1 = i0 + 1
    j1 = j0 + 1
    if i1 > nx + 2:
        i1 = nx + 2
    if j1 > ny + 1:
        j1 = ny + 1

    v00 = u[i0, j0]
    v10 = u[i1, j0]
    v01 = u[i0, j1]
    v11 = u[i1, j1]

    a = v00 * (1.0 - tx) + v10 * tx
    b = v01 * (1.0 - tx) + v11 * tx
    return a * (1.0 - ty) + b * ty


@njit(inline="always")
def sample_v(v, x, y, x_min, y_min, dx, dy, nx, ny):
    fx = (x - x_min) / dx + 0.5
    fy = (y - y_min) / dy

    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0

    i0 = clamp(i0, 0, nx)
    j0 = clamp(j0, 0, ny + 1)

    i1 = i0 + 1
    j1 = j0 + 1
    if i1 > nx + 1:
        i1 = nx + 1
    if j1 > ny + 2:
        j1 = ny + 2

    v00 = v[i0, j0]
    v10 = v[i1, j0]
    v01 = v[i0, j1]
    v11 = v[i1, j1]

    a = v00 * (1.0 - tx) + v10 * tx
    b = v01 * (1.0 - tx) + v11 * tx
    return a * (1.0 - ty) + b * ty


@njit(inline="always")
def vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny):
    return (
        sample_u(u, x, y, x_min, y_min, dx, dy, nx, ny),
        sample_v(v, x, y, x_min, y_min, dx, dy, nx, ny),
    )


@njit(inline="always")
def sample_p(p, x, y, x_min, y_min, dx, dy, nx, ny):
    fx = (x - x_min) / dx + 0.5
    fy = (y - y_min) / dy + 0.5

    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0

    if i0 < 0:
        i0 = 0
    if i0 > nx:
        i0 = nx
    if j0 < 0:
        j0 = 0
    if j0 > ny:
        j0 = ny

    i1 = i0 + 1
    j1 = j0 + 1
    if i1 > nx + 1:
        i1 = nx + 1
    if j1 > ny + 1:
        j1 = ny + 1

    v00 = p[i0, j0]
    v10 = p[i1, j0]
    v01 = p[i0, j1]
    v11 = p[i1, j1]

    a = v00 * (1.0 - tx) + v10 * tx
    b = v01 * (1.0 - tx) + v11 * tx
    return a * (1.0 - ty) + b * ty


# ============================================================
# Pressure BC (Neumann) + gauge fix
# ============================================================

@njit
def apply_bc_pressure_neumann(p, nx, ny):
    for j in range(0, ny + 2):
        p[0, j] = p[1, j]
        p[nx + 1, j] = p[nx, j]
    for i in range(0, nx + 2):
        p[i, 0] = p[i, 1]
        p[i, ny + 1] = p[i, ny]
    p[1, 1] = 0.0


# ============================================================
# BCs - Airfoil domain
# ============================================================

@njit
def apply_bc_airfoil(u, v, p, nx, ny, Ux, Uy):
    apply_bc_pressure_neumann(p, nx, ny)

    # u inlet/top/bottom, outlet grad=0
    for j in range(1, ny + 1):
        u[1, j] = Ux
        u[0, j] = 2.0 * Ux - u[1, j]
        u[nx + 1, j] = u[nx, j]
        u[nx + 2, j] = u[nx + 1, j]

    for i in range(1, nx + 2):
        u[i, 0] = 2.0 * Ux - u[i, 1]
        u[i, ny + 1] = 2.0 * Ux - u[i, ny]

    # v inlet
    for j in range(1, ny + 2):
        v[1, j] = Uy
        v[0, j] = 2.0 * Uy - v[1, j]
        v[nx + 1, j] = v[nx, j]

    # v topo/baixo
    for i in range(1, nx + 1):
        v[i, 1] = Uy
        v[i, 0] = 2.0 * Uy - v[i, 1]
        v[i, ny + 1] = Uy
        v[i, ny + 2] = 2.0 * Uy - v[i, ny + 1]


# ============================================================
# Advection (semi-Lagrangian RK2)
# ============================================================

@njit(parallel=True)
def advect_semi_lagrangian(u0, v0, u1, v1, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny):
    for j in prange(1, ny + 1):
        y = y_min + (j - 0.5) * dy
        for i in range(1, nx + 2):
            x = x_min + i * dx

            uvel, vvel = vel_at(u0, v0, x, y, x_min, y_min, dx, dy, nx, ny)
            xm = x - 0.5 * dt * uvel
            ym = y - 0.5 * dt * vvel
            uvel2, vvel2 = vel_at(u0, v0, xm, ym, x_min, y_min, dx, dy, nx, ny)

            xb = x - dt * uvel2
            yb = y - dt * vvel2
            xb = clamp(xb, x_min, x_max)
            yb = clamp(yb, y_min, y_max)

            u1[i, j] = sample_u(u0, xb, yb, x_min, y_min, dx, dy, nx, ny)

    for j in prange(1, ny + 2):
        y = y_min + j * dy
        for i in range(1, nx + 1):
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
# Diffusion (explicit Laplacian)
# ============================================================

@njit(parallel=True)
def diffuse_inplace(u_in, v_in, u_out, v_out, nu, dt, dx, dy, nx, ny):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    for j in prange(1, ny + 1):
        for i in range(1, nx + 2):
            lap = (u_in[i + 1, j] - 2 * u_in[i, j] + u_in[i - 1, j]) * idx2 + \
                  (u_in[i, j + 1] - 2 * u_in[i, j] + u_in[i, j - 1]) * idy2
            u_out[i, j] = u_in[i, j] + nu * dt * lap

    for j in prange(1, ny + 2):
        for i in range(1, nx + 1):
            lap = (v_in[i + 1, j] - 2 * v_in[i, j] + v_in[i - 1, j]) * idx2 + \
                  (v_in[i, j + 1] - 2 * v_in[i, j] + v_in[i, j - 1]) * idy2
            v_out[i, j] = v_in[i, j] + nu * dt * lap


# ============================================================
# Brinkman penalization
# ============================================================

@njit(parallel=True)
def brinkman_penalize(u, v, chi_u, chi_v, dt, eta, nx, ny):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 2):
            chi = chi_u[i - 1, j - 1]
            if chi > 1e-8:
                u[i, j] = u[i, j] / (1.0 + dt * chi / eta)

    for j in prange(1, ny + 2):
        for i in range(1, nx + 1):
            chi = chi_v[i - 1, j - 1]
            if chi > 1e-8:
                v[i, j] = v[i, j] / (1.0 + dt * chi / eta)


# ============================================================
# Divergence + Projection
# ============================================================

@njit(parallel=True)
def compute_divergence(u, v, div, dx, dy, nx, ny):
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            div[i, j] = (u[i + 1, j] - u[i, j]) * invdx + (v[i, j + 1] - v[i, j]) * invdy


@njit(parallel=True)
def project(u, v, p, dt, rho, dx, dy, nx, ny):
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    scale = dt / rho

    for j in prange(1, ny + 1):
        for i in range(1, nx + 2):
            u[i, j] -= scale * (p[i, j] - p[i - 1, j]) * invdx

    for j in prange(1, ny + 2):
        for i in range(1, nx + 1):
            v[i, j] -= scale * (p[i, j] - p[i, j - 1]) * invdy


# ============================================================
# Poisson PCG (fallback)
# ============================================================

@njit(parallel=True)
def apply_A(p, Ap, dx, dy, nx, ny):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            Ap[i, j] = (p[i + 1, j] - 2 * p[i, j] + p[i - 1, j]) * idx2 + \
                       (p[i, j + 1] - 2 * p[i, j] + p[i, j - 1]) * idy2


@njit
def dot_inner(a, b, nx, ny):
    s = 0.0
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            s += a[i, j] * b[i, j]
    return s


@njit
def zero_mean_rhs_inplace(b, nx, ny):
    s = 0.0
    n = nx * ny
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            s += b[i, j]
    mean_b = s / n
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            b[i, j] -= mean_b


@njit
def pcg_poisson_inplace(p, b, r, z, d, Ap, dx, dy, nx, ny, max_iter=800, tol=1e-6):
    apply_bc_pressure_neumann(p, nx, ny)

    apply_A(p, Ap, dx, dy, nx, ny)
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            r[i, j] = b[i, j] - Ap[i, j]

    diag = -2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy))
    invdiag = 1.0 / diag

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            z[i, j] = r[i, j] * invdiag
            d[i, j] = z[i, j]

    rz_old = dot_inner(r, z, nx, ny)
    bnorm = np.sqrt(max(dot_inner(b, b, nx, ny), 1e-30))

    for it in range(max_iter):
        apply_bc_pressure_neumann(p, nx, ny)

        apply_A(d, Ap, dx, dy, nx, ny)
        denom = dot_inner(d, Ap, nx, ny)
        if abs(denom) < 1e-30:
            return it + 1

        alpha = rz_old / denom

        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                p[i, j] += alpha * d[i, j]
                r[i, j] -= alpha * Ap[i, j]

        rnorm = np.sqrt(dot_inner(r, r, nx, ny)) / bnorm
        if rnorm < tol:
            apply_bc_pressure_neumann(p, nx, ny)
            return it + 1

        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                z[i, j] = r[i, j] * invdiag

        rz_new = dot_inner(r, z, nx, ny)
        beta = rz_new / rz_old
        rz_old = rz_new

        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                d[i, j] = z[i, j] + beta * d[i, j]

    apply_bc_pressure_neumann(p, nx, ny)
    return max_iter


# ============================================================
# Multigrid Poisson (FAST default)
# ============================================================

@njit(parallel=True)
def mg_relax_rb_gs(p, b, dx, dy, nx, ny, omega=1.0, iters=2):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    diag = -2.0 * (idx2 + idy2)

    for _ in range(iters):
        for color in range(2):
            for j in prange(1, ny + 1):
                for i in range(1, nx + 1):
                    if ((i + j) & 1) != color:
                        continue
                    off = (p[i + 1, j] + p[i - 1, j]) * idx2 + (p[i, j + 1] + p[i, j - 1]) * idy2
                    p_new = (b[i, j] - off) / diag
                    p[i, j] = (1.0 - omega) * p[i, j] + omega * p_new

        apply_bc_pressure_neumann(p, nx, ny)
        p[1, 1] = 0.0


@njit(parallel=True)
def mg_residual(p, b, r, dx, dy, nx, ny):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            Ap = (p[i + 1, j] - 2.0 * p[i, j] + p[i - 1, j]) * idx2 + \
                 (p[i, j + 1] - 2.0 * p[i, j] + p[i, j - 1]) * idy2
            r[i, j] = b[i, j] - Ap


@njit(parallel=True)
def mg_restrict_fullweight(r_f, b_c, nxf, nyf):
    nxc = nxf // 2
    nyc = nyf // 2
    for jc in prange(1, nyc + 1):
        jf = 2 * jc
        for ic in range(1, nxc + 1):
            if_ = 2 * ic
            b_c[ic, jc] = (
                4.0 * r_f[if_, jf] +
                2.0 * (r_f[if_ - 1, jf] + r_f[if_ + 1, jf] + r_f[if_, jf - 1] + r_f[if_, jf + 1]) +
                (r_f[if_ - 1, jf - 1] + r_f[if_ - 1, jf + 1] + r_f[if_ + 1, jf - 1] + r_f[if_ + 1, jf + 1])
            ) / 16.0


@njit(parallel=True)
def mg_prolong_bilinear(p_c, e_f, nxf, nyf):
    nxc = nxf // 2
    nyc = nyf // 2

    for j in prange(0, nyf + 2):
        for i in range(0, nxf + 2):
            e_f[i, j] = 0.0

    for jc in prange(1, nyc + 1):
        for ic in range(1, nxc + 1):
            ef_i = 2 * ic
            ef_j = 2 * jc

            v00 = p_c[ic, jc]
            v10 = p_c[ic + 1, jc] if ic < nxc else v00
            v01 = p_c[ic, jc + 1] if jc < nyc else v00
            v11 = p_c[ic + 1, jc + 1] if (ic < nxc and jc < nyc) else v00

            e_f[ef_i, ef_j] += v00
            if ef_i + 1 <= nxf:
                e_f[ef_i + 1, ef_j] += 0.5 * (v00 + v10)
            if ef_j + 1 <= nyf:
                e_f[ef_i, ef_j + 1] += 0.5 * (v00 + v01)
            if (ef_i + 1 <= nxf) and (ef_j + 1 <= nyf):
                e_f[ef_i + 1, ef_j + 1] += 0.25 * (v00 + v10 + v01 + v11)


def _build_mg_context(nx, ny, dx, dy, dtype=np.float32, min_size=32):
    # Finest level placeholders will be overwritten with actual p,b refs.
    Ps = [None]
    Bs = [None]
    Rs = []
    Es = []
    dxs = [float(dx)]
    dys = [float(dy)]
    nxs = [int(nx)]
    nys = [int(ny)]

    while (nxs[-1] > min_size and nys[-1] > min_size and
           (nxs[-1] % 2 == 0) and (nys[-1] % 2 == 0)):
        nxc = nxs[-1] // 2
        nyc = nys[-1] // 2
        Ps.append(np.zeros((nxc + 2, nyc + 2), dtype=dtype))
        Bs.append(np.zeros((nxc + 2, nyc + 2), dtype=dtype))
        dxs.append(dxs[-1] * 2.0)
        dys.append(dys[-1] * 2.0)
        nxs.append(nxc)
        nys.append(nyc)

    # residual/correction buffers
    for lev in range(len(Ps)):
        # allocate based on each level size
        if lev == 0:
            # placeholders (will be replaced by finest size after binding)
            Rs.append(None)
            Es.append(None)
        else:
            Rs.append(np.zeros_like(Ps[lev]))
            Es.append(np.zeros_like(Ps[lev]))

    return {"Ps": Ps, "Bs": Bs, "Rs": Rs, "Es": Es, "dxs": dxs, "dys": dys, "nxs": nxs, "nys": nys}


def mg_poisson_inplace(p_f, b_f, mgctx, vcycles=10, pre=2, post=2, omega=1.0, coarse_relax=60):
    """
    In-place V-cycle multigrid for Laplace(p)=b with Neumann BC.
    mgctx is cached per (nx,ny,dx,dy).
    Returns number of vcycles executed.
    """
    Ps = mgctx["Ps"]
    Bs = mgctx["Bs"]
    Rs = mgctx["Rs"]
    Es = mgctx["Es"]
    dxs = mgctx["dxs"]
    dys = mgctx["dys"]
    nxs = mgctx["nxs"]
    nys = mgctx["nys"]

    # bind finest arrays
    Ps[0] = p_f
    Bs[0] = b_f
    if Rs[0] is None or Rs[0].shape != p_f.shape:
        Rs[0] = np.zeros_like(p_f)
    if Es[0] is None or Es[0].shape != p_f.shape:
        Es[0] = np.zeros_like(p_f)

    levels = len(Ps)

    for _ in range(int(vcycles)):
        # down
        for lev in range(levels - 1):
            mg_relax_rb_gs(Ps[lev], Bs[lev], dxs[lev], dys[lev], nxs[lev], nys[lev], omega=omega, iters=pre)

            mg_residual(Ps[lev], Bs[lev], Rs[lev], dxs[lev], dys[lev], nxs[lev], nys[lev])
            apply_bc_pressure_neumann(Rs[lev], nxs[lev], nys[lev])

            Bs[lev + 1][:, :] = 0.0
            mg_restrict_fullweight(Rs[lev], Bs[lev + 1], nxs[lev], nys[lev])
            apply_bc_pressure_neumann(Bs[lev + 1], nxs[lev + 1], nys[lev + 1])
            Bs[lev + 1][1, 1] = 0.0

            Ps[lev + 1][:, :] = 0.0
            apply_bc_pressure_neumann(Ps[lev + 1], nxs[lev + 1], nys[lev + 1])
            Ps[lev + 1][1, 1] = 0.0

        # coarse
        last = levels - 1
        mg_relax_rb_gs(Ps[last], Bs[last], dxs[last], dys[last], nxs[last], nys[last], omega=omega, iters=coarse_relax)

        # up
        for lev in range(levels - 2, -1, -1):
            mg_prolong_bilinear(Ps[lev + 1], Es[lev], nxs[lev], nys[lev])

            Ps[lev][1:nxs[lev] + 1, 1:nys[lev] + 1] += Es[lev][1:nxs[lev] + 1, 1:nys[lev] + 1]
            apply_bc_pressure_neumann(Ps[lev], nxs[lev], nys[lev])
            Ps[lev][1, 1] = 0.0

            mg_relax_rb_gs(Ps[lev], Bs[lev], dxs[lev], dys[lev], nxs[lev], nys[lev], omega=omega, iters=post)

    return int(vcycles)


# ============================================================
# Forces: Brinkman
# ============================================================

def compute_force_coeffs_brinkman(u, v, chi_u, chi_v, rho, eta, Uinf, alpha_rad, dx, dy, nx, ny):
    Fx = rho * (dx * dy / eta) * float(np.sum(chi_u * u[1:nx + 2, 1:ny + 1]))
    Fy = rho * (dx * dy / eta) * float(np.sum(chi_v * v[1:nx + 1, 1:ny + 2]))

    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    D = Fx * ca + Fy * sa
    L = -Fx * sa + Fy * ca

    q = 0.5 * rho * Uinf * Uinf
    Cd = D / (q + 1e-30)
    Cl = L / (q + 1e-30)
    return float(Cl), float(Cd)


# ============================================================
# Forces: Control Volume (FIXED dt_eff)
# prev_mom = (mx, my, step_index)
# ============================================================

@njit
def cv_integrate_momentum(u, v, rho, x_min, y_min, dx, dy, nx, ny, x1, x2, y1, y2):
    mx = 0.0
    my = 0.0
    vol = dx * dy

    for j in range(1, ny + 1):
        yc = y_min + (j - 0.5) * dy
        if yc < y1 or yc > y2:
            continue
        for i in range(1, nx + 1):
            xc = x_min + (i - 0.5) * dx
            if xc < x1 or xc > x2:
                continue

            uc = 0.5 * (u[i, j] + u[i + 1, j])
            vc = 0.5 * (v[i, j] + v[i, j + 1])

            mx += rho * uc * vol
            my += rho * vc * vol

    return mx, my


@njit
def cv_surface_forces(u, v, p, rho, x_min, y_min, dx, dy, nx, ny, x1, x2, y1, y2):
    Cx = 0.0
    Cy = 0.0
    Px = 0.0
    Py = 0.0

    nseg_y = max(8, int(np.round((y2 - y1) / dy)) * 2)
    nseg_x = max(8, int(np.round((x2 - x1) / dx)) * 2)
    ds_y = (y2 - y1) / nseg_y
    ds_x = (x2 - x1) / nseg_x

    # Left face x=x1, n=(-1,0)
    x = x1
    nxn = -1.0
    nyn = 0.0
    for k in range(nseg_y):
        y = y1 + (k + 0.5) * ds_y
        ux, vy = vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny)
        un = ux * nxn + vy * nyn
        Cx += rho * ux * un * ds_y
        Cy += rho * vy * un * ds_y

        pp = sample_p(p, x, y, x_min, y_min, dx, dy, nx, ny)
        Px += -(pp * nxn) * ds_y
        Py += -(pp * nyn) * ds_y

    # Right face x=x2, n=(+1,0)
    x = x2
    nxn = 1.0
    nyn = 0.0
    for k in range(nseg_y):
        y = y1 + (k + 0.5) * ds_y
        ux, vy = vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny)
        un = ux * nxn + vy * nyn
        Cx += rho * ux * un * ds_y
        Cy += rho * vy * un * ds_y

        pp = sample_p(p, x, y, x_min, y_min, dx, dy, nx, ny)
        Px += -(pp * nxn) * ds_y
        Py += -(pp * nyn) * ds_y

    # Bottom face y=y1, n=(0,-1)
    y = y1
    nxn = 0.0
    nyn = -1.0
    for k in range(nseg_x):
        x = x1 + (k + 0.5) * ds_x
        ux, vy = vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny)
        un = ux * nxn + vy * nyn
        Cx += rho * ux * un * ds_x
        Cy += rho * vy * un * ds_x

        pp = sample_p(p, x, y, x_min, y_min, dx, dy, nx, ny)
        Px += -(pp * nxn) * ds_x
        Py += -(pp * nyn) * ds_x

    # Top face y=y2, n=(0,+1)
    y = y2
    nxn = 0.0
    nyn = 1.0
    for k in range(nseg_x):
        x = x1 + (k + 0.5) * ds_x
        ux, vy = vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny)
        un = ux * nxn + vy * nyn
        Cx += rho * ux * un * ds_x
        Cy += rho * vy * un * ds_x

        pp = sample_p(p, x, y, x_min, y_min, dx, dy, nx, ny)
        Px += -(pp * nxn) * ds_x
        Py += -(pp * nyn) * ds_x

    return Cx, Cy, Px, Py


def compute_force_coeffs_control_volume(u, v, p, rho, dt_base, step_idx,
                                        x_min, y_min, dx, dy, nx, ny,
                                        Uinf, alpha_rad,
                                        cv_box, prev_mom=None):
    x1, x2, y1, y2 = cv_box

    mx, my = cv_integrate_momentum(u, v, rho, x_min, y_min, dx, dy, nx, ny, x1, x2, y1, y2)

    if prev_mom is None:
        dmxdt = 0.0
        dmydt = 0.0
    else:
        pmx, pmy, pstep = prev_mom
        nstep = max(1, int(step_idx) - int(pstep))
        dt_eff = float(dt_base) * float(nstep)
        dmxdt = (mx - pmx) / max(dt_eff, 1e-30)
        dmydt = (my - pmy) / max(dt_eff, 1e-30)

    Cx, Cy, Px, Py = cv_surface_forces(u, v, p, rho, x_min, y_min, dx, dy, nx, ny, x1, x2, y1, y2)

    Fx = -dmxdt - Cx + Px
    Fy = -dmydt - Cy + Py

    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    D = Fx * ca + Fy * sa
    L = -Fx * sa + Fy * ca

    q = 0.5 * rho * Uinf * Uinf
    Cd = D / (q + 1e-30)
    Cl = L / (q + 1e-30)

    new_prev = (float(mx), float(my), int(step_idx))
    return float(Cl), float(Cd), (float(Fx), float(Fy)), new_prev


# ============================================================
# Outputs: cell-centered, vorticity, particles
# ============================================================

@njit(parallel=True)
def make_cell_centered(u, v, uc, vc, nx, ny):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            uc[i - 1, j - 1] = 0.5 * (u[i, j] + u[i + 1, j])
            vc[i - 1, j - 1] = 0.5 * (v[i, j] + v[i, j + 1])


@njit(parallel=True)
def compute_vorticity(uc, vc, w, dx, dy, nx, ny):
    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            dvdx = (vc[i + 1, j] - vc[i - 1, j]) * inv2dx
            dudy = (uc[i, j + 1] - uc[i, j - 1]) * inv2dy
            w[i, j] = dvdx - dudy


@njit(parallel=True)
def advect_particles_rk2(xp, yp, u, v, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny):
    for k in prange(xp.shape[0]):
        x = xp[k]
        y = yp[k]

        u1, v1 = vel_at(u, v, x, y, x_min, y_min, dx, dy, nx, ny)
        xm = x + 0.5 * dt * u1
        ym = y + 0.5 * dt * v1
        u2, v2 = vel_at(u, v, xm, ym, x_min, y_min, dx, dy, nx, ny)

        xn = x + dt * u2
        yn = y + dt * v2

        if xn < x_min:
            xn = x_min
        if xn > x_max:
            xn = x_max
        if yn < y_min:
            yn = y_min
        if yn > y_max:
            yn = y_max

        xp[k] = xn
        yp[k] = yn


# ============================================================
# Public API
# ============================================================

def simulate_airfoil_naca4412(
    nx, ny, steps, dt,
    rho, nu,
    Uinf, alpha_deg,
    x_min, x_max, y_min, y_max,
    chi_cell, eta=1e-3,
    out_every=200,

    init_state=None,

    stop_enable=True,
    stop_min_steps=600,
    stop_check_every=100,
    stop_window=4,
    stop_tol_cl=2e-3,
    stop_tol_div=5e-4,

    force_method="cv",     # "brinkman" | "cv" | "both"
    cv_box=(-0.25, 1.25, -0.50, 0.50),

    # Pressure solver selection
    pressure_solver="mg",  # "mg" (fast) | "pcg" (fallback)

    # MG params
    mg_vcycles=10,
    mg_pre=2,
    mg_post=2,
    mg_omega=1.0,
    mg_coarse_relax=60,
    mg_min_size=32,

    # PCG params
    pcg_max_iter=800,
    pcg_tol=1e-6,

    record_every=0,
    record_max=0,
    with_particles=False,
    n_particles=0,
    seed_x=-1.5,
    seed_ymin=-0.4,
    seed_ymax=0.4,
    particles_seed=1234,
):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    u = np.zeros((nx + 3, ny + 2), dtype=np.float32)
    v = np.zeros((nx + 2, ny + 3), dtype=np.float32)
    p = np.zeros((nx + 2, ny + 2), dtype=np.float32)

    u_adv = np.zeros_like(u)
    v_adv = np.zeros_like(v)
    u_star = np.zeros_like(u)
    v_star = np.zeros_like(v)

    div = np.zeros_like(p)
    b = np.zeros_like(p)

    # PCG buffers (still allocated; cheap vs runtime)
    r = np.zeros_like(p)
    z = np.zeros_like(p)
    dvec = np.zeros_like(p)
    Ap = np.zeros_like(p)

    alpha = np.deg2rad(alpha_deg)
    Ux = Uinf * np.cos(alpha)
    Uy = Uinf * np.sin(alpha)

    chi_cell = np.asarray(chi_cell, dtype=np.float32)

    chi_u = np.zeros((nx + 1, ny), dtype=np.float32)
    for j in range(ny):
        for i in range(nx + 1):
            left = chi_cell[i - 1, j] if i - 1 >= 0 else 0.0
            right = chi_cell[i, j] if i < nx else 0.0
            chi_u[i, j] = 0.5 * (left + right)

    chi_v = np.zeros((nx, ny + 1), dtype=np.float32)
    for j in range(ny + 1):
        for i in range(nx):
            bot = chi_cell[i, j - 1] if j - 1 >= 0 else 0.0
            top = chi_cell[i, j] if j < ny else 0.0
            chi_v[i, j] = 0.5 * (bot + top)

    # init
    if isinstance(init_state, dict) and ("u" in init_state) and ("v" in init_state) and ("p" in init_state):
        u[:, :] = np.asarray(init_state["u"], dtype=np.float32)
        v[:, :] = np.asarray(init_state["v"], dtype=np.float32)
        p[:, :] = np.asarray(init_state["p"], dtype=np.float32)
    else:
        u[1:nx + 2, 1:ny + 1] = Ux
        v[1:nx + 1, 1:ny + 2] = Uy
        p[:, :] = 0.0

    apply_bc_airfoil(u, v, p, nx, ny, Ux, Uy)

    # MG context (cached once)
    mgctx = None
    if str(pressure_solver).lower() == "mg":
        mgctx = _build_mg_context(nx, ny, dx, dy, dtype=np.float32, min_size=int(mg_min_size))

    # recording
    frames_w = []
    particles_frames = None

    uc_buf = vc_buf = w_buf = None
    if record_every and record_max and record_max > 0:
        uc_buf = np.zeros((nx, ny), dtype=np.float32)
        vc_buf = np.zeros((nx, ny), dtype=np.float32)
        w_buf = np.zeros((nx, ny), dtype=np.float32)

    xp = yp = None
    if with_particles and (n_particles is not None) and (n_particles > 0):
        rng = np.random.default_rng(int(particles_seed))
        xp = np.full((int(n_particles),), float(seed_x), dtype=np.float32)
        yp = rng.uniform(float(seed_ymin), float(seed_ymax), size=(int(n_particles),)).astype(np.float32)
        particles_frames = []

    # early stop
    cl_hist = []
    div_hist = []
    last_check_step = 0
    stopped_early = False

    prev_mom_cv = None

    Cl = Cd = 0.0
    Cl_b = Cd_b = 0.0
    Cl_cv = Cd_cv = 0.0

    solver_tag = "mg" if str(pressure_solver).lower() == "mg" else "pcg"

    for n in range(1, steps + 1):
        advect_semi_lagrangian(u, v, u_adv, v_adv, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny)
        diffuse_inplace(u_adv, v_adv, u_star, v_star, nu, dt, dx, dy, nx, ny)
        brinkman_penalize(u_star, v_star, chi_u, chi_v, dt, eta, nx, ny)
        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)

        compute_divergence(u_star, v_star, div, dx, dy, nx, ny)
        b[1:nx + 1, 1:ny + 1] = (rho / dt) * div[1:nx + 1, 1:ny + 1]
        zero_mean_rhs_inplace(b, nx, ny)

        if solver_tag == "mg":
            iters = mg_poisson_inplace(
                p, b, mgctx,
                vcycles=int(mg_vcycles),
                pre=int(mg_pre),
                post=int(mg_post),
                omega=float(mg_omega),
                coarse_relax=int(mg_coarse_relax),
            )
        else:
            iters = pcg_poisson_inplace(p, b, r, z, dvec, Ap, dx, dy, nx, ny, max_iter=int(pcg_max_iter), tol=float(pcg_tol))

        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)

        project(u_star, v_star, p, dt, rho, dx, dy, nx, ny)
        apply_bc_airfoil(u_star, v_star, p, nx, ny, Ux, Uy)

        u, u_star = u_star, u
        v, v_star = v_star, v

        if xp is not None:
            advect_particles_rk2(xp, yp, u, v, dt, x_min, x_max, y_min, y_max, dx, dy, nx, ny)

        if record_every and record_max and (n % int(record_every) == 0) and (len(frames_w) < int(record_max)):
            make_cell_centered(u, v, uc_buf, vc_buf, nx, ny)
            w_buf[:, :] = 0.0
            compute_vorticity(uc_buf, vc_buf, w_buf, dx, dy, nx, ny)
            frames_w.append(w_buf.astype(np.float16).copy())

            if particles_frames is not None:
                P = np.empty((xp.shape[0], 2), dtype=np.float32)
                P[:, 0] = xp
                P[:, 1] = yp
                particles_frames.append(P)

        if out_every and (n % int(out_every) == 0):
            dnorm = float(np.sqrt(np.mean(div[1:nx + 1, 1:ny + 1] ** 2)))

            Cl_b, Cd_b = compute_force_coeffs_brinkman(u, v, chi_u, chi_v, rho, eta, Uinf, alpha, dx, dy, nx, ny)
            Cl_cv, Cd_cv, _, prev_mom_cv = compute_force_coeffs_control_volume(
                u, v, p, rho, dt, n,
                x_min, y_min, dx, dy, nx, ny,
                Uinf, alpha, cv_box, prev_mom=prev_mom_cv
            )

            if force_method == "brinkman":
                Cl, Cd = Cl_b, Cd_b
                print(f"[airfoil] step {n}/{steps} | {solver_tag} {iters} | div_rms {dnorm:.3e} | Cl~{Cl:.3f} Cd~{Cd:.3f}")
            elif force_method == "cv":
                Cl, Cd = Cl_cv, Cd_cv
                print(f"[airfoil] step {n}/{steps} | {solver_tag} {iters} | div_rms {dnorm:.3e} | Cl~{Cl:.3f} Cd~{Cd:.3f}")
            else:
                Cl, Cd = Cl_cv, Cd_cv
                print(
                    f"[airfoil] step {n}/{steps} | {solver_tag} {iters} | div_rms {dnorm:.3e} | "
                    f"Brinkman Cl~{Cl_b:.3f} Cd~{Cd_b:.3f} | "
                    f"CV Cl~{Cl_cv:.3f} Cd~{Cd_cv:.3f}"
                )

        if stop_enable and (n >= int(stop_min_steps)) and (stop_check_every > 0) and (n - last_check_step >= int(stop_check_every)):
            last_check_step = n
            dnorm = float(np.sqrt(np.mean(div[1:nx + 1, 1:ny + 1] ** 2)))

            Cl_b, Cd_b = compute_force_coeffs_brinkman(u, v, chi_u, chi_v, rho, eta, Uinf, alpha, dx, dy, nx, ny)
            Cl_cv, Cd_cv, _, prev_mom_cv = compute_force_coeffs_control_volume(
                u, v, p, rho, dt, n,
                x_min, y_min, dx, dy, nx, ny,
                Uinf, alpha, cv_box, prev_mom=prev_mom_cv
            )

            Cl_now = Cl_b if force_method == "brinkman" else Cl_cv

            cl_hist.append(float(Cl_now))
            div_hist.append(float(dnorm))
            if len(cl_hist) > int(stop_window):
                cl_hist.pop(0)
                div_hist.pop(0)

            if len(cl_hist) == int(stop_window):
                cl_range = max(cl_hist) - min(cl_hist)
                div_range = max(div_hist) - min(div_hist)
                if (cl_range < float(stop_tol_cl)) and (div_range < float(stop_tol_div)):
                    stopped_early = True
                    break

    # final forces
    Cl_b, Cd_b = compute_force_coeffs_brinkman(u, v, chi_u, chi_v, rho, eta, Uinf, alpha, dx, dy, nx, ny)
    Cl_cv, Cd_cv, _, _ = compute_force_coeffs_control_volume(
        u, v, p, rho, dt, n,
        x_min, y_min, dx, dy, nx, ny,
        Uinf, alpha, cv_box, prev_mom=None
    )

    if force_method == "brinkman":
        Cl, Cd = Cl_b, Cd_b
    else:
        Cl, Cd = Cl_cv, Cd_cv

    uc = np.zeros((nx, ny), dtype=np.float32)
    vc = np.zeros((nx, ny), dtype=np.float32)
    make_cell_centered(u, v, uc, vc, nx, ny)

    return {
        "p": p[1:nx + 1, 1:ny + 1].copy(),
        "uc": uc,
        "vc": vc,

        "Cl": float(Cl),
        "Cd": float(Cd),

        "Cl_brinkman": float(Cl_b),
        "Cd_brinkman": float(Cd_b),
        "Cl_cv": float(Cl_cv),
        "Cd_cv": float(Cd_cv),
        "cv_box": tuple(cv_box),
        "force_method": str(force_method),

        "dx": float(dx),
        "dy": float(dy),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),

        "state": {"u": u.copy(), "v": v.copy(), "p": p.copy()},

        "steps_ran": int(n),
        "stopped_early": bool(stopped_early),

        "frames_w": frames_w,
        "particles": particles_frames,
        "record_every": int(record_every) if record_every else 0,
        "pressure_solver": solver_tag,
    }
