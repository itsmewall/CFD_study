# run.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geometry import naca4_coordinates, build_chi_cell
from mac_solver import simulate_airfoil_naca4412


STATE_PATH = "state_last.npz"


def load_state(path=STATE_PATH):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return {"u": data["u"], "v": data["v"], "p": data["p"]}
    except Exception:
        return None


def save_state(state, path=STATE_PATH):
    try:
        np.savez_compressed(path, u=state["u"], v=state["v"], p=state["p"])
    except Exception:
        pass


def pick_dt(Uinf, dx, dy, cfl=0.6, dt_cap=0.0020):
    dt = cfl * min(dx, dy) / max(Uinf, 1e-9)
    return float(min(dt, dt_cap))


def main():
    # =========================
    # Config (boa qualidade x custo)
    # =========================
    nx, ny = 320, 160
    steps = 2800
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

    # Aerofólio NACA 4412
    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=900)

    # >>> FIX CRÍTICO: flip no Y para corrigir o sinal de Cl(0) do 4412
    FLIP_AIRFOIL_Y = True
    if FLIP_AIRFOIL_Y:
        Y = -Y

    # chi fracionário + suavização
    chi_cell, dx, dy = build_chi_cell(
        nx, ny, x_min, x_max, y_min, y_max, X, Y,
        supersample=5,
        sdf_smooth=True,
        eps_cells=2.0
    )

    dt = pick_dt(Uinf, dx, dy, cfl=0.6, dt_cap=0.0020)
    eta = 1e-3

    # CV box mais afastada (menos viés/ruído)
    cv_box = (-1.0 * c, 2.5 * c, -1.2 * c, 1.2 * c)

    # Visualização temporal
    record_every = 25
    record_max = 240
    with_particles = True
    n_particles = 3500

    # Warm start
    init_state = load_state()

    # Pressão: use PCG como default (robusto). MG só se quiser testar.
    pressure_solver = "pcg"  # "pcg" (recomendado) | "mg"
    mg_fallback_rtol = 0.15  # se usar mg, cai pra pcg automaticamente se residual estiver ruim

    print("=== Config ===")
    print(f"grid={nx}x{ny}  dx={dx:.5f} dy={dy:.5f}  dt={dt:.5f}")
    print(f"Re={Re} alpha={alpha_deg}deg  steps={steps}")
    print(f"cv_box={cv_box}  pressure_solver={pressure_solver}")
    print(f"flip_airfoil_y={FLIP_AIRFOIL_Y}")
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

        init_state=init_state,

        force_method="both",
        cv_box=cv_box,

        pressure_solver=pressure_solver,
        mg_fallback_rtol=mg_fallback_rtol,

        mg_vcycles=10,
        mg_pre=2,
        mg_post=2,
        mg_omega=1.0,
        mg_coarse_relax=60,
        mg_min_size=32,

        pcg_max_iter=600,
        pcg_tol=1e-6,

        stop_enable=True,
        stop_min_steps=900,
        stop_check_every=100,
        stop_window=4,
        stop_tol_cl=2e-3,
        stop_tol_div=5e-4,

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

    print(f"\nTempo total: {t1 - t0:.2f}s | steps_ran={res['steps_ran']} stopped_early={res['stopped_early']}")
    print(f"Cl~{res['Cl']:.3f} Cd~{res['Cd']:.3f} | solver={res.get('pressure_solver','?')}")
    print(f"Brinkman: Cl~{res.get('Cl_brinkman', np.nan):.3f} Cd~{res.get('Cd_brinkman', np.nan):.3f}")
    print(f"CV      : Cl~{res.get('Cl_cv', np.nan):.3f} Cd~{res.get('Cd_cv', np.nan):.3f}")

    if not (np.isfinite(res["Cl"]) and np.isfinite(res["Cd"])):
        print("\n[ERRO] Cl/Cd não finitos. (Se estiver em MG, use PCG.)")
        return

    # salva estado
    save_state(res["state"])

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

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
        plt.tight_layout()
        plt.show()

    # =========================
    # Salvar resultado
    # =========================
    np.savez_compressed(
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
        flip_airfoil_y=FLIP_AIRFOIL_Y,
    )


if __name__ == "__main__":
    main()
