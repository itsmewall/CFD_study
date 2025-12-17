import numpy as np
from numba import njit

# ============================================================
# NACA 4 dígitos (corda=1)
# ============================================================

def naca4_coordinates(m: float, p: float, t: float, n: int = 700):
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))  # cos spacing

    yt = 5.0 * t * (
        0.2969 * np.sqrt(x + 1e-12)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    for i in range(n):
        xi = x[i]
        if xi < p:
            yc[i] = m / (p**2) * (2*p*xi - xi**2)
            dyc[i] = 2*m / (p**2) * (p - xi)
        else:
            yc[i] = m / ((1-p)**2) * ((1 - 2*p) + 2*p*xi - xi**2)
            dyc[i] = 2*m / ((1-p)**2) * (p - xi)

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # upper TE->LE + lower LE->TE
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y

def rotate_translate(X, Y, alpha_rad: float, x0: float, y0: float):
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    xr = ca * X - sa * Y
    yr = sa * X + ca * Y
    return xr + x0, yr + y0

# ============================================================
# Numba helpers: point-in-poly e distância ao polígono
# ============================================================

@njit
def _point_in_poly(x, y, X, Y):
    inside = False
    n = X.shape[0]
    xj = X[n-1]
    yj = Y[n-1]
    for i in range(n):
        xi = X[i]
        yi = Y[i]
        # Ray casting
        cond = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        if cond:
            inside = not inside
        xj = xi
        yj = yi
    return inside

@njit
def _dist_point_segment_sq(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx*abx + aby*aby + 1e-30
    t = (apx*abx + apy*aby) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t*abx
    cy = ay + t*aby
    dx = px - cx
    dy = py - cy
    return dx*dx + dy*dy

@njit
def _min_dist_to_poly(px, py, X, Y):
    n = X.shape[0]
    dmin = 1e30
    xj = X[n-1]
    yj = Y[n-1]
    for i in range(n):
        xi = X[i]
        yi = Y[i]
        d2 = _dist_point_segment_sq(px, py, xj, yj, xi, yi)
        if d2 < dmin:
            dmin = d2
        xj = xi
        yj = yi
    return np.sqrt(dmin)

@njit
def _clamp01(a):
    if a < 0.0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a

@njit
def _build_chi(nx, ny, x_min, x_max, y_min, y_max, X, Y, supersample, use_sdf_smooth, eps_cells):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    chi = np.zeros((nx, ny), dtype=np.float32)

    ss = supersample
    inv_ss2 = 1.0 / (ss * ss)

    eps = eps_cells * (dx if dx < dy else dy)

    for i in range(nx):
        cx = x_min + (i + 0.5) * dx
        for j in range(ny):
            cy = y_min + (j + 0.5) * dy

            # --- Super-sampling: fração sólida na célula
            inside_count = 0
            for a in range(ss):
                for b in range(ss):
                    # ponto dentro da célula
                    px = x_min + (i + (a + 0.5) / ss) * dx
                    py = y_min + (j + (b + 0.5) / ss) * dy
                    if _point_in_poly(px, py, X, Y):
                        inside_count += 1
            frac = inside_count * inv_ss2  # 0..1

            if use_sdf_smooth:
                # --- SDF no centro da célula: signed distance
                inside_center = _point_in_poly(cx, cy, X, Y)
                dist = _min_dist_to_poly(cx, cy, X, Y)
                sdf = -dist if inside_center else dist

                # transição suave numa faixa eps
                chi_sdf = _clamp01(0.5 - sdf / (eps + 1e-30))

                # combina frac (volume fraction) com a suavização do contorno
                # (mantém fração, mas reduz serrilhado)
                val = 0.5 * (frac + chi_sdf)
            else:
                val = frac

            chi[i, j] = np.float32(_clamp01(val))

    return chi, dx, dy

def build_chi_cell(nx, ny, x_min, x_max, y_min, y_max, polyX, polyY,
                   supersample: int = 4,
                   sdf_smooth: bool = True,
                   eps_cells: float = 2.0):
    """
    Retorna:
      chi_cell: (nx,ny) float32 em [0,1] (fração sólida + suavização opcional)
      dx, dy
    """
    X = np.asarray(polyX, dtype=np.float64)
    Y = np.asarray(polyY, dtype=np.float64)
    chi, dx, dy = _build_chi(nx, ny, x_min, x_max, y_min, y_max, X, Y,
                             supersample, sdf_smooth, eps_cells)
    return chi, dx, dy
