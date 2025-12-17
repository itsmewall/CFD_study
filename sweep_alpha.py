import time
import numpy as np
import matplotlib.pyplot as plt

from geometry import naca4_coordinates, build_chi_cell
from mac_solver import simulate_airfoil_naca4412


def main():
    # =========================
    # Config do sweep
    # =========================
    nx, ny = 320, 160
    steps_max = 3000           # teto; o early stop corta antes
    out_every = 0              # sem spam no sweep

    Uinf = 1.0
    Re = 2000.0
    rho = 1.0
    nu = Uinf / Re
    dt = 0.0015
    eta = 1e-3

    c = 1.0
    x_min, x_max = -2.0 * c, 4.0 * c
    y_min, y_max = -1.5 * c, 1.5 * c

    # Range de alpha
    alphas = np.arange(-4.0, 13.0, 2.0)   # -4..12 step 2
    alphas = np.asarray(alphas, dtype=float)

    # Estratégia: varrer em ordem crescente (continuação)
    alphas_sorted = np.sort(alphas)

    # Early stop (ajuste fino aqui)
    stop_min_steps = 700
    stop_check_every = 100
    stop_window = 4
    stop_tol_cl = 2e-3
    stop_tol_div = 5e-4

    # =========================
    # Geometria + máscara (1x)
    # =========================
    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=700)

    print("[sweep] Gerando chi_cell (uma vez)...")
    chi_cell, _, _ = build_chi_cell(
        nx, ny, x_min, x_max, y_min, y_max, X, Y,
        supersample=4,
        sdf_smooth=True,
        eps_cells=2.0
    )

    # =========================
    # Rodar sweep com warm start
    # =========================
    Cl_map = {}
    Cd_map = {}
    steps_map = {}
    stopped_map = {}

    prev_state = None
    t0 = time.time()

    for a in alphas_sorted:
        print(f"[sweep] alpha={a:.1f} deg ...")
        res = simulate_airfoil_naca4412(
            nx, ny, steps_max, dt,
            rho, nu,
            Uinf, float(a),
            x_min, x_max, y_min, y_max,
            chi_cell,
            eta=eta,
            out_every=out_every,

            # warm start
            init_state=prev_state,

            # early stop
            stop_enable=True,
            stop_min_steps=stop_min_steps,
            stop_check_every=stop_check_every,
            stop_window=stop_window,
            stop_tol_cl=stop_tol_cl,
            stop_tol_div=stop_tol_div,

            # desligar gravações no sweep
            record_every=0,
            record_max=0,
            with_particles=False,
            n_particles=0
        )

        Cl_map[a] = res["Cl"]
        Cd_map[a] = res["Cd"]
        steps_map[a] = res["steps_ran"]
        stopped_map[a] = res["stopped_early"]

        prev_state = res["state"]  # warm start para o próximo alpha

        flag = "STOP" if res["stopped_early"] else "MAX"
        print(f"    -> Cl={res['Cl']:.4f} Cd={res['Cd']:.4f} | steps={res['steps_ran']} ({flag})")

    t1 = time.time()
    print(f"[sweep] Tempo total: {t1 - t0:.2f}s")

    # Remontar na ordem original (para plot / CSV)
    Cl = np.array([Cl_map[a] for a in alphas], dtype=float)
    Cd = np.array([Cd_map[a] for a in alphas], dtype=float)
    steps_ran = np.array([steps_map[a] for a in alphas], dtype=int)
    stopped = np.array([1 if stopped_map[a] else 0 for a in alphas], dtype=int)

    # Salvar
    data = np.column_stack([alphas, Cl, Cd, steps_ran, stopped])
    np.savetxt(
        "polar_CL_CD_vs_alpha.csv",
        data,
        delimiter=",",
        header="alpha_deg,Cl,Cd,steps_ran,stopped_early(1/0)",
        comments=""
    )
    np.savez("polar_CL_CD_vs_alpha.npz", alpha_deg=alphas, Cl=Cl, Cd=Cd, steps_ran=steps_ran, stopped=stopped)

    # Plot Cl x alpha
    plt.figure()
    plt.plot(alphas, Cl, marker="o")
    plt.xlabel("alpha (deg)")
    plt.ylabel("Cl")
    plt.title("NACA 4412: Cl vs alpha (warm start + early stop)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Cd x alpha (opcional)
    plt.figure()
    plt.plot(alphas, Cd, marker="o")
    plt.xlabel("alpha (deg)")
    plt.ylabel("Cd")
    plt.title("NACA 4412: Cd vs alpha (warm start + early stop)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
