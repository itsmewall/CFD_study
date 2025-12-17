import time
import numpy as np

from geometry import naca4_coordinates, build_chi_cell
from mac_solver import simulate_airfoil_naca4412


def main():
    nx, ny = 320, 160
    steps_max = 2800
    dt = 0.0015

    Uinf = 1.0
    Re = 2000.0
    rho = 1.0
    nu = Uinf / Re
    eta = 1e-3

    c = 1.0
    x_min, x_max = -2.0 * c, 4.0 * c
    y_min, y_max = -1.5 * c, 1.5 * c

    # Caixa do control volume (em torno do aerofólio)
    cv_box = (-0.75, 2.0, -0.9, 0.9)

    # ângulos para comparar
    alphas = [0.0, 4.0, 10.0]

    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=700)
    chi_cell, _, _ = build_chi_cell(
        nx, ny, x_min, x_max, y_min, y_max, X, Y,
        supersample=4, sdf_smooth=True, eps_cells=2.0
    )

    t0 = time.time()
    prev_state = None

    for a in alphas:
        print(f"\n=== alpha={a:.1f} deg ===")
        res = simulate_airfoil_naca4412(
            nx, ny, steps_max, dt,
            rho, nu,
            Uinf, a,
            x_min, x_max, y_min, y_max,
            chi_cell,
            eta=eta,
            out_every=250,

            # warm start ajuda até aqui
            init_state=prev_state,

            # early stop do item 1
            stop_enable=True,
            stop_min_steps=700,
            stop_check_every=100,
            stop_window=4,
            stop_tol_cl=2e-3,
            stop_tol_div=5e-4,

            # novo item 2
            force_method="both",
            cv_box=cv_box,

            # sem gravações
            record_every=0,
            with_particles=False,
        )

        prev_state = res["state"]

        print(f"Brinkman: Cl={res['Cl_brinkman']:.4f}  Cd={res['Cd_brinkman']:.4f}")
        print(f"CV      : Cl={res['Cl_cv']:.4f}  Cd={res['Cd_cv']:.4f}")
        print(f"steps_ran={res['steps_ran']} stopped_early={res['stopped_early']}")

    t1 = time.time()
    print(f"\nTempo total: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
