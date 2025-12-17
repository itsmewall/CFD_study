import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geometry import naca4_coordinates, build_chi_cell
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

    dt = 0.0015
    eta = 1e-3

    # Domínio
    c = 1.0
    x_min, x_max = -2.0 * c, 4.0 * c
    y_min, y_max = -1.5 * c, 1.5 * c

    # Aerofólio NACA 4412 (fixo; AoA entra no freestream do solver)
    X, Y = naca4_coordinates(0.04, 0.4, 0.12, n=700)

    # chi fracionário + suavização (pré-processamento)
    chi_cell, dx, dy = build_chi_cell(
        nx, ny, x_min, x_max, y_min, y_max, X, Y,
        supersample=4,
        sdf_smooth=True,
        eps_cells=2.0
    )

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
        # gravação temporal:
        record_every=25,
        record_max=240,
        with_particles=True,
        n_particles=4000,
        seed_x=-1.5,
        seed_ymin=-0.4,
        seed_ymax=0.4
    )
    t1 = time.time()

    print(f"Tempo total: {t1 - t0:.2f}s | Cl~{res['Cl']:.3f} Cd~{res['Cd']:.3f}")

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
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.tight_layout()
    plt.show()

    # =========================
    # Animação: vorticidade + tracers
    # =========================
    frames = res["frames_w"]
    parts = res["particles"]

    if not frames:
        print("Sem frames gravados (record_every/record_max).")
        return

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

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.6, 0.6)

    sc = None
    if parts is not None and len(parts) > 0:
        sc = ax.scatter(parts[0][:, 0], parts[0][:, 1], s=1)

    def update(k):
        im.set_data(frames[k].T)
        if sc is not None:
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

    # Salvar resultado bruto
    np.savez(
        "naca4412_timeviz.npz",
        p=res["p"], uc=uc, vc=vc,
        Cl=res["Cl"], Cd=res["Cd"],
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        dx=res["dx"], dy=res["dy"],
        record_every=res["record_every"],
    )


if __name__ == "__main__":
    main()
