import numpy as np

def naca4_coordinates(m: float, p: float, t: float, n: int = 600):
    """
    Polígono fechado do NACA 4 dígitos (corda = 1).
    Retorna (X, Y) em sentido fechado.
    """
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))  # cos spacing (melhor no bordo de ataque)

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
            yc[i] = m / (p**2) * (2 * p * xi - xi**2)
            dyc[i] = 2 * m / (p**2) * (p - xi)
        else:
            yc[i] = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xi - xi**2)
            dyc[i] = 2 * m / ((1 - p) ** 2) * (p - xi)

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

def points_in_poly(px, py, X, Y):
    """
    Ray casting vetorizado.
    px,py: arrays 2D
    X,Y: vertices 1D do polígono fechado implicitamente
    """
    inside = np.zeros(px.shape, dtype=bool)
    n = len(X)
    xj, yj = X[-1], Y[-1]
    for i in range(n):
        xi, yi = X[i], Y[i]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        xj, yj = xi, yi
    return inside

def build_chi_cell(nx, ny, x_min, x_max, y_min, y_max, polyX, polyY):
    """
    chi_cell (nx,ny) em centros de célula: 1 dentro do aerofólio, 0 fora.
    """
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    xc = x_min + (np.arange(nx) + 0.5) * dx
    yc = y_min + (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")  # (nx,ny)
    inside = points_in_poly(Xc, Yc, polyX, polyY)
    return inside.astype(np.float32), dx, dy
