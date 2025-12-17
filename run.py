import time
import numpy as np
import matplotlib.pyplot as plt

from geometry import naca4_coordinates, rotate_translate, build_chi_cell
from mac_solver import simulate_airfoil_naca4412


def main():
    # =========================
    # Config pro seu PC
    # =========================
    nx, ny = 320, 160
    steps = 2500
    out_every = 250

    # Escoamento
    Uinf = 1.0
    alpha_deg = 4.0
    Re = 2000.0
    rho = 1.0
    nu = Uinf / Re

    # Melhor estabilidade / fidelidade
    dt = 0.0015
    eta = 1e-3

    # Domínio
    c = 1.0
    x_min, x_max = -2.0 * c, 4.0 * c
    y_min, y_max = -1.5 * c, 1.5 * c

    # Aerofólio NACA 4412: m=0.04, p=0.4, t=0.12
    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=700)

    # Aerofólio FIXO (não rotaciona o corpo); o AoA entra no Ux,Uy dentro do solver
    Xr, Yr = X, Y

    # chi fracionário + suavização (pré-processamento; custo quase todo aqui, não no solver)
    chi_cell, dx, dy = build_chi_cell(
        nx, ny, x_min, x_max, y_min, y_max, Xr, Yr,
        supersample=4,
        sdf_smooth=True,
        eps_cells=2.0
    )

    t0 = time.time()
    res = simulate_airfoil_naca4412(
        nx, ny, steps, dt,
        rho, nu,
        Uinf, alpha_deg,
        x_min, x_max, y_min, y_max,
        chi_cell,
        eta=eta,
        out_every=out_every
    )
    t1 = time.time()

    print(f"Tempo total: {t1 - t0:.2f}s | Cl~{res['Cl']:.3f} Cd~{res['Cd']:.3f}")

    # =========================
    # Visualizações melhores
    # =========================
    uc = res["uc"]
    vc = res["vc"]
    speed = np.sqrt(uc**2 + vc**2)

    # 1) Velocidade (zoom)
    plt.figure(figsize=(10, 4))
    plt.title("Velocidade |u| (zoom)")
    plt.imshow(
        speed.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="equal",
        interpolation="bilinear",
    )
    plt.colorbar()
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.tight_layout()
    plt.show()

    # 2) Vorticidade (muito melhor pra “ver” a física)
    dx = res["dx"]
    dy = res["dy"]
    w = (np.roll(vc, -1, axis=0) - np.roll(vc, 1, axis=0)) / (2 * dx) - \
        (np.roll(uc, -1, axis=1) - np.roll(uc, 1, axis=1)) / (2 * dy)

    plt.figure(figsize=(10, 4))
    plt.title("Vorticidade (zoom)")
    plt.imshow(
        w.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="equal",
        interpolation="bilinear",
    )
    plt.colorbar()
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.tight_layout()
    plt.show()

    # Salvar
    np.savez(
        "naca4412_final.npz",
        p=res["p"], uc=uc, vc=vc,
        Cl=res["Cl"], Cd=res["Cd"],
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        dx=res["dx"], dy=res["dy"]
    )


if __name__ == "__main__":
    main()
