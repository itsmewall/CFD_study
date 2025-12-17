import time
import numpy as np
import matplotlib.pyplot as plt

from geometry import naca4_coordinates, rotate_translate, build_chi_cell
from mac_solver import simulate_airfoil_naca4412

def main():
    # =========================
    # Config rápida pro seu PC
    # =========================
    nx, ny = 320, 160          # comece aqui; depois 480x240 se aguentar
    steps = 2500               # aumente para “assentar”
    out_every = 250

    # Escoamento
    Uinf = 1.0
    alpha_deg = 4.0
    Re = 2000.0
    rho = 1.0
    nu = Uinf / Re

    dt = 0.0025                # v1 estável (semi-lagrangiano ajuda)
    eta = 1e-3                 # penalização (parede “dura”)

    # Domínio (menor para performance; suficiente p/ v1)
    c = 1.0
    x_min, x_max = -2.0*c, 4.0*c
    y_min, y_max = -1.5*c, 1.5*c

    # Aerofólio NACA 4412: m=0.04, p=0.4, t=0.12
    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=700)

    # Posiciona no domínio: LE em x=0, corda 0..1; centraliza em y=0
    # (pivô em (0,0) -> rotaciona em torno do LE; ok para v1)
    Xr, Yr = rotate_translate(X, Y, np.deg2rad(alpha_deg), x0=0.0, y0=0.0)

    chi_cell, dx, dy = build_chi_cell(nx, ny, x_min, x_max, y_min, y_max, Xr, Yr)

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

    # Visualização
    speed = np.sqrt(res["uc"]**2 + res["vc"]**2)

    plt.figure()
    plt.title("Velocidade (magnitude) - cell centered")
    plt.imshow(speed.T, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Salvar
    np.savez("naca4412_final.npz", p=res["p"], uc=res["uc"], vc=res["vc"],
             Cl=res["Cl"], Cd=res["Cd"],
             x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, dx=res["dx"], dy=res["dy"])

if __name__ == "__main__":
    main()
