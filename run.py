# run.py
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geometry import naca4_coordinates, build_chi_cell
from mac_solver import simulate_airfoil_naca4412


def _pick_dt(Uinf, dx, dy, cfl=0.6, dt_cap=0.0025):
    # semi-Lagrangian é estável, mas dt grande piora forças; isso dá um “top” seguro
    dt = cfl * min(dx, dy) / max(Uinf, 1e-9)
    return float(min(dt, dt_cap))


def main():
    # =========================
    # Config pro seu PC (boa qualidade x custo)
    # =========================
    nx, ny = 320, 160
    steps = 2800                 # dá mais assentamento
    out_every = 250

    # Escoamento
    Uinf = 1.0
    alpha_deg = 4.0
    Re = 2000.0
    rho = 1.0
    nu = Uinf / Re

    # Domínio
    c = 1.0
    x_min, x_max = -2.0 * c, 4.0 * c
    y_min, y_max = -1.5 * c, 1.5 * c

    # Aerofólio NACA 4412 (AoA entra no freestream do solver)
    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=900)

    # chi fracionário + suavização
    # Obs: supersample alto custa só no pré-processamento (não encarece o solver)
    chi_cell, dx, dy = build_chi_cell(
        nx, ny, x_min, x_max, y_min, y_max, X, Y,
        supersample=5,
        sdf_smooth=True,
        eps_cells=2.0
    )

    # Time-step: escolha automática decente (forças mais estáveis)
    dt = _pick_dt(Uinf, dx, dy, cfl=0.6, dt_cap=0.0020)

    # Brinkman: parede “dura” mas sem explodir numericamente
    eta = 1e-3

    # CV box: mais longe do aerofólio reduz viés (CV sem termo viscoso)
    cv_box = (-0.75 * c, 2.0 * c, -0.9 * c, 0.9 * c)

    # Time-viz (vorticidade + partículas)
    record_every = 25
    record_max = 240
    with_particles = True
    n_particles = 3500

    print("=== Config ===")
    print(f"grid={nx}x{ny}  dx={dx:.5f} dy={dy:.5f}  dt={dt:.5f}")
    print(f"Re={Re} alpha={alpha_deg}deg  steps={steps}")
    print(f"cv_box={cv_box}")
    print("================\n")

    # =========================
    # Rodar simulação
    # =========================
    t0 = time.time()
    res = simulate_airfoil_naca4412(
        nx, ny, steps, dt,
        rho, nu,
        Uinf, alpha_deg,
        x_min, x_max, y_min, y_max,
        chi_cell,
        eta=eta,
        out_every=out_every,

        # Forças: padrão confiável (Brinkman). CV fica como sanity-check.
        force_method="both",
        cv_box=cv_box,

        # Aceleração real: multigrid no Poisson (seu mac_solver já suporta)
        pressure_solver="mg",
        mg_vcycles=10,
        mg_pre=2,
        mg_post=2,
        mg_omega=1.0,
        mg_coarse_relax=60,
        mg_min_size=32,

        # Early stop (para economizar tempo quando estabiliza)
        stop_enable=True,
        stop_min_steps=900,
        stop_check_every=100,
        stop_window=4,
        stop_tol_cl=2e-3,
        stop_tol_div=5e-4,

        # gravação temporal
        record_every=record_every,
        record_max=record_max,
        with_particles=with_particles,
        n_particles=n_particles,
        seed_x=-1.5 * c,
        seed_ymin=-0.45 * c,
        seed_ymax=0.45 * c,
        particles_seed=1234,
    )
    t1 = time.time()

    print(f"\nTempo total: {t1 - t0:.2f}s | Cl~{res['Cl']:.3f} Cd~{res['Cd']:.3f}")
    if "pressure_solver" in res:
        print(f"Pressure solver: {res['pressure_solver']}")

    # =========================
    # Plot final (zoom)
    # =========================
    uc = res["uc"]
    vc = res["vc"]
    speed = np.sqrt(uc**2 + vc**2)

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
    plt.xlim(-0.5 * c, 2.5 * c)
    plt.ylim(-0.6 * c, 0.6 * c)
    plt.tight_layout()
    plt.show()

    # =========================
    # Animação: vorticidade + tracers
    # =========================
    frames = res.get("frames_w", [])
    parts = res.get("particles", None)

    if not frames:
        print("Sem frames gravados (record_every/record_max).")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Vorticidade ω(t) + tracers")

        im = ax.imshow(
            frames[0].T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
            interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax)

        ax.set_xlim(-0.5 * c, 2.5 * c)
        ax.set_ylim(-0.6 * c, 0.6 * c)

        sc = None
        if parts is not None and len(parts) > 0:
            sc = ax.scatter(parts[0][:, 0], parts[0][:, 1], s=1)

        def update(k):
            im.set_data(frames[k].T)
            if sc is not None and parts is not None and k < len(parts):
                sc.set_offsets(parts[k])
                return im, sc
            return (im,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=40, blit=True
        )
        plt.tight_layout()
        plt.show()

        # Salvar (opcional; requer ffmpeg para mp4)
        # ani.save("vorticidade_tracers.mp4", fps=25, dpi=140)

    # =========================
    # Salvar resultado bruto
    # =========================
    np.savez(
        "naca4412_timeviz.npz",
        p=res["p"], uc=uc, vc=vc,
        Cl=res["Cl"], Cd=res["Cd"],
        Cl_brinkman=res.get("Cl_brinkman", np.nan),
        Cd_brinkman=res.get("Cd_brinkman", np.nan),
        Cl_cv=res.get("Cl_cv", np.nan),
        Cd_cv=res.get("Cd_cv", np.nan),
        cv_box=np.array(res.get("cv_box", cv_box), dtype=np.float32),
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        dx=res["dx"], dy=res["dy"],
        record_every=res.get("record_every", 0),
        pressure_solver=res.get("pressure_solver", "unknown"),
    )


if __name__ == "__main__":
    main()
