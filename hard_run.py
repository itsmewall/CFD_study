# hard_run.py
# Rodada “hard” + relatório: sweeps, sensitividades, métricas de estabilidade/performance e recomendações.
# Saídas:
#   out/<timestamp>/
#     results.csv
#     config.json
#     report.md
#     cl_alpha.png, cd_alpha.png, polar.png
#     dt_sensitivity.png
#     grid_sensitivity.png
#     states/ (warm-start por caso)

import os
import json
import time
import math
import csv
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
import inspect

import numpy as np
import matplotlib.pyplot as plt

from geometry import naca4_coordinates, build_chi_cell
from mac_solver import simulate_airfoil_naca4412


# ============================================================
# Util
# ============================================================

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def pick_dt(Uinf, dx, dy, cfl=0.60, dt_cap=0.0020):
    dt = cfl * min(dx, dy) / max(Uinf, 1e-9)
    return float(min(dt, dt_cap))


def safe_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def save_state_npz(state, path):
    try:
        np.savez_compressed(path, u=state["u"], v=state["v"], p=state["p"])
        return True
    except Exception:
        return False


def load_state_npz(path):
    try:
        d = np.load(path, allow_pickle=True)
        return {"u": d["u"], "v": d["v"], "p": d["p"]}
    except Exception:
        return None


# ============================================================
# Métricas “preciosas” (pós-sim)
# ============================================================

def divergence_rms_from_state(state, dx, dy, nx, ny):
    u = state["u"]
    v = state["v"]
    # células 1..nx,1..ny no arranjo do solver
    div = (u[2:nx+2, 1:ny+1] - u[1:nx+1, 1:ny+1]) / dx + (v[1:nx+1, 2:ny+2] - v[1:nx+1, 1:ny+1]) / dy
    return float(np.sqrt(np.mean(div.astype(np.float64) ** 2)))


def vorticity_stats(uc, vc, dx, dy):
    # uc, vc são cell-centered (nx, ny)
    # w = dv/dx - du/dy (diferença central interna)
    uc64 = uc.astype(np.float64)
    vc64 = vc.astype(np.float64)

    dvdx = (vc64[2:, 1:-1] - vc64[:-2, 1:-1]) / (2.0 * dx)
    dudy = (uc64[1:-1, 2:] - uc64[1:-1, :-2]) / (2.0 * dy)
    w = dvdx - dudy

    w_abs = np.abs(w)
    w2 = w * w

    return {
        "w_rms": float(np.sqrt(np.mean(w2))),
        "w_max": float(np.max(w_abs)),
        "enstrophy": float(0.5 * np.mean(w2)),
    }


def flow_stats(uc, vc):
    sp = np.sqrt(uc.astype(np.float64) ** 2 + vc.astype(np.float64) ** 2)
    return {
        "speed_mean": float(np.mean(sp)),
        "speed_max": float(np.max(sp)),
        "ke_mean": float(0.5 * np.mean(sp * sp)),
    }


def linear_fit_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    # R²
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return float(m), float(r2)


# ============================================================
# Config
# ============================================================

@dataclass
class BaseConfig:
    # Escoamento
    Uinf: float = 1.0
    rho: float = 1.0
    alpha_deg_baseline: float = 4.0
    Re_baseline: float = 2000.0

    # Brinkman
    eta: float = 1e-3

    # Domínio
    c: float = 1.0
    x_min: float = -2.0
    x_max: float = 4.0
    y_min: float = -1.5
    y_max: float = 1.5

    # Geometria
    naca_m: float = 0.04
    naca_p: float = 0.4
    naca_t: float = 0.12
    naca_pts: int = 900

    # Malha imersa (pré-processamento)
    supersample: int = 5
    sdf_smooth: bool = True
    eps_cells: float = 2.0

    # Solver run
    steps: int = 2800
    out_every: int = 250

    # Early stop (economiza tempo)
    stop_enable: bool = True
    stop_min_steps: int = 900
    stop_check_every: int = 100
    stop_window: int = 4
    stop_tol_cl: float = 2e-3
    stop_tol_div: float = 5e-4

    # Forças
    force_method: str = "both"  # "cv" | "brinkman" | "both"
    cv_box: tuple = (-0.75, 2.0, -0.9, 0.9)

    # Pressão (default PCG; MG só se você corrigiu o NaN do prolongamento)
    pressure_solver: str = "pcg"  # "pcg" | "mg"
    # PCG
    pcg_max_iter: int = 600
    pcg_tol: float = 1e-6
    # MG
    mg_vcycles: int = 10
    mg_pre: int = 2
    mg_post: int = 2
    mg_omega: float = 1.0
    mg_coarse_relax: int = 60
    mg_min_size: int = 32


# ============================================================
# Runner
# ============================================================

def build_chi_cached(cache, nx, ny, dom, geo, pre):
    key = (nx, ny, dom["x_min"], dom["x_max"], dom["y_min"], dom["y_max"],
           geo["m"], geo["p"], geo["t"], geo["npts"],
           pre["supersample"], pre["sdf_smooth"], pre["eps_cells"])
    if key in cache:
        return cache[key]

    X, Y = naca4_coordinates(geo["m"], geo["p"], geo["t"], n=geo["npts"])
    chi_cell, dx, dy = build_chi_cell(
        nx, ny, dom["x_min"], dom["x_max"], dom["y_min"], dom["y_max"], X, Y,
        supersample=pre["supersample"],
        sdf_smooth=pre["sdf_smooth"],
        eps_cells=pre["eps_cells"],
    )
    cache[key] = (chi_cell, dx, dy, X, Y)
    return cache[key]


def run_one_case(case, cfg: BaseConfig, chi_pack, out_dir, warm_state_path=None):
    chi_cell, dx, dy, _, _ = chi_pack
    nx, ny = case["nx"], case["ny"]

    Uinf = cfg.Uinf
    rho = cfg.rho
    Re = case["Re"]
    nu = Uinf / Re

    dom = case["dom"]
    dt0 = pick_dt(Uinf, dx, dy, cfl=case.get("cfl", 0.60), dt_cap=case.get("dt_cap", 0.0020))
    dt = float(dt0 * case.get("dt_factor", 1.0))

    init_state = load_state_npz(warm_state_path) if warm_state_path else None

    kwargs = dict(
        eta=cfg.eta,
        out_every=cfg.out_every,

        init_state=init_state,

        force_method=cfg.force_method,
        cv_box=case.get("cv_box", cfg.cv_box),

        pressure_solver=case.get("pressure_solver", cfg.pressure_solver),

        # MG
        mg_vcycles=cfg.mg_vcycles,
        mg_pre=cfg.mg_pre,
        mg_post=cfg.mg_post,
        mg_omega=cfg.mg_omega,
        mg_coarse_relax=cfg.mg_coarse_relax,
        mg_min_size=cfg.mg_min_size,

        # PCG
        pcg_max_iter=cfg.pcg_max_iter,
        pcg_tol=cfg.pcg_tol,

        # early stop
        stop_enable=cfg.stop_enable,
        stop_min_steps=cfg.stop_min_steps,
        stop_check_every=cfg.stop_check_every,
        stop_window=cfg.stop_window,
        stop_tol_cl=cfg.stop_tol_cl,
        stop_tol_div=cfg.stop_tol_div,

        # sem gravação temporal aqui (economia de tempo)
        record_every=0,
        record_max=0,
        with_particles=False,
        n_particles=0,
    )
    kwargs = safe_kwargs(simulate_airfoil_naca4412, kwargs)

    t0 = time.perf_counter()
    res = simulate_airfoil_naca4412(
        nx, ny, cfg.steps, dt,
        rho, nu,
        Uinf, case["alpha_deg"],
        dom["x_min"], dom["x_max"], dom["y_min"], dom["y_max"],
        chi_cell,
        **kwargs
    )
    t1 = time.perf_counter()

    # métricas pós
    state = res.get("state", None)
    div_rms = float("nan")
    if state is not None:
        div_rms = divergence_rms_from_state(state, res["dx"], res["dy"], nx, ny)

    uc = res["uc"]
    vc = res["vc"]
    vst = vorticity_stats(uc, vc, res["dx"], res["dy"])
    fst = flow_stats(uc, vc)

    # salvar state para warm-start
    saved_state = False
    if state is not None:
        sp = ensure_dir(os.path.join(out_dir, "states"))
        saved_state = save_state_npz(state, os.path.join(sp, f"{case['case_id']}.npz"))

    # consolidar
    row = {
        "case_id": case["case_id"],
        "tag": case.get("tag", ""),
        "nx": nx,
        "ny": ny,
        "Re": float(Re),
        "alpha_deg": float(case["alpha_deg"]),
        "dt": float(dt),
        "pressure_solver": str(res.get("pressure_solver", case.get("pressure_solver", cfg.pressure_solver))),
        "steps_ran": int(res.get("steps_ran", cfg.steps)),
        "stopped_early": bool(res.get("stopped_early", False)),

        "Cl": float(res["Cl"]),
        "Cd": float(res["Cd"]),
        "Cl_brinkman": float(res.get("Cl_brinkman", np.nan)),
        "Cd_brinkman": float(res.get("Cd_brinkman", np.nan)),
        "Cl_cv": float(res.get("Cl_cv", np.nan)),
        "Cd_cv": float(res.get("Cd_cv", np.nan)),

        "div_rms_post": float(div_rms),

        "speed_mean": float(fst["speed_mean"]),
        "speed_max": float(fst["speed_max"]),
        "ke_mean": float(fst["ke_mean"]),

        "w_rms": float(vst["w_rms"]),
        "w_max": float(vst["w_max"]),
        "enstrophy": float(vst["enstrophy"]),

        "runtime_s": float(t1 - t0),
        "s_per_step": float((t1 - t0) / max(int(res.get("steps_ran", cfg.steps)), 1)),
        "warm_state_loaded": bool(init_state is not None),
        "warm_state_saved": bool(saved_state),
    }
    return row


def write_csv_header(path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_csv_row(path, fieldnames, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()


# ============================================================
# Relatório
# ============================================================

def plot_sweep(out_dir, rows, tag_filter, title_prefix):
    R = [r for r in rows if r.get("tag", "") == tag_filter and np.isfinite(r["Cl"]) and np.isfinite(r["Cd"])]
    if not R:
        return

    al = np.array([r["alpha_deg"] for r in R], dtype=np.float64)
    Cl = np.array([r["Cl"] for r in R], dtype=np.float64)
    Cd = np.array([r["Cd"] for r in R], dtype=np.float64)

    idx = np.argsort(al)
    al, Cl, Cd = al[idx], Cl[idx], Cd[idx]

    # Cl x alpha
    plt.figure(figsize=(9, 4))
    plt.title(f"{title_prefix} — Cl x alpha")
    plt.plot(al, Cl, marker="o")
    plt.xlabel("alpha (deg)")
    plt.ylabel("Cl")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cl_alpha.png"), dpi=160)
    plt.close()

    # Cd x alpha
    plt.figure(figsize=(9, 4))
    plt.title(f"{title_prefix} — Cd x alpha")
    plt.plot(al, Cd, marker="o")
    plt.xlabel("alpha (deg)")
    plt.ylabel("Cd")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cd_alpha.png"), dpi=160)
    plt.close()

    # Polar
    plt.figure(figsize=(5.2, 5.2))
    plt.title(f"{title_prefix} — Polar (Cd x Cl)")
    plt.plot(Cd, Cl, marker="o")
    plt.xlabel("Cd")
    plt.ylabel("Cl")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "polar.png"), dpi=160)
    plt.close()


def plot_sensitivity(out_dir, rows, tag, xkey, ykeys, fname, title):
    R = [r for r in rows if r.get("tag", "") == tag and np.isfinite(r["Cl"]) and np.isfinite(r["Cd"])]
    if not R:
        return
    x = np.array([r[xkey] for r in R], dtype=np.float64)

    plt.figure(figsize=(9, 4))
    plt.title(title)
    for yk in ykeys:
        y = np.array([r[yk] for r in R], dtype=np.float64)
        plt.plot(x, y, marker="o", label=yk)
    plt.xlabel(xkey)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def make_report(out_dir, cfg: BaseConfig, rows):
    # foco: baseline sweep
    base = [r for r in rows if r.get("tag", "") == "sweep_alpha_Re2000"]
    base_ok = [r for r in base if np.isfinite(r["Cl"]) and np.isfinite(r["Cd"])]

    lines = []
    lines.append("# Relatório — hard_run\n")
    lines.append(f"- Data: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
    lines.append(f"- Runs totais: {len(rows)} (ok: {sum(1 for r in rows if np.isfinite(r['Cl']) and np.isfinite(r['Cd']))})\n")
    lines.append(f"- Config base: steps={cfg.steps}, out_every={cfg.out_every}, eta={cfg.eta}, force_method={cfg.force_method}\n")

    # performance geral
    runt = np.array([r["runtime_s"] for r in rows if np.isfinite(r["runtime_s"])], dtype=np.float64)
    sps = np.array([r["s_per_step"] for r in rows if np.isfinite(r["s_per_step"])], dtype=np.float64)
    if runt.size:
        lines.append("\n## Performance\n")
        lines.append(f"- Tempo médio por caso: {np.mean(runt):.2f}s (min {np.min(runt):.2f}s, max {np.max(runt):.2f}s)\n")
        lines.append(f"- Tempo médio por step: {np.mean(sps):.6f}s/step\n")

    # baseline slope
    if base_ok:
        al_deg = np.array([r["alpha_deg"] for r in base_ok], dtype=np.float64)
        Cl = np.array([r["Cl"] for r in base_ok], dtype=np.float64)

        # usa só faixa “linear” (ex.: -2..6)
        mask = (al_deg >= -2.0) & (al_deg <= 6.0)
        m, r2 = linear_fit_slope(np.deg2rad(al_deg[mask]), Cl[mask]) if np.any(mask) else (float("nan"), float("nan"))

        lines.append("\n## Sweep alpha (Re=2000)\n")
        lines.append(f"- Pontos: {len(base_ok)}\n")
        if np.isfinite(m):
            lines.append(f"- Inclinação Cl_alpha (faixa -2..6 deg): {m:.3f} por rad (R²={r2:.3f})\n")

        # divergência
        divs = np.array([r["div_rms_post"] for r in base_ok], dtype=np.float64)
        lines.append(f"- div_rms pós (médio): {np.mean(divs):.3e} | (pior): {np.max(divs):.3e}\n")

        # discrepância Brinkman vs CV (se existir)
        if np.any(np.isfinite([r["Cl_brinkman"] for r in base_ok])) and np.any(np.isfinite([r["Cl_cv"] for r in base_ok])):
            dCl = []
            dCd = []
            for r in base_ok:
                if np.isfinite(r["Cl_brinkman"]) and np.isfinite(r["Cl_cv"]):
                    dCl.append(abs(r["Cl_brinkman"] - r["Cl_cv"]))
                if np.isfinite(r["Cd_brinkman"]) and np.isfinite(r["Cd_cv"]):
                    dCd.append(abs(r["Cd_brinkman"] - r["Cd_cv"]))
            if dCl:
                lines.append(f"- |ΔCl| médio (Brinkman vs CV): {np.mean(dCl):.3f}\n")
            if dCd:
                lines.append(f"- |ΔCd| médio (Brinkman vs CV): {np.mean(dCd):.3f}\n")

    # recomendações automáticas (pragmáticas)
    lines.append("\n## Diagnóstico e recomendações (automático)\n")
    rec = []

    # NaNs
    n_bad = sum(1 for r in rows if not (np.isfinite(r["Cl"]) and np.isfinite(r["Cd"])))
    if n_bad:
        rec.append(f"- Há {n_bad} casos com NaN/Inf. Se pressão_solver='mg', isso quase sempre é **prolongamento com race**. Use PCG ou corrija mg_prolong_bilinear (sem parallel/sem += concorrente).")

    # divergência
    worst_div = max((r["div_rms_post"] for r in rows if np.isfinite(r["div_rms_post"])), default=float("nan"))
    if np.isfinite(worst_div) and worst_div > 5e-2:
        rec.append(f"- div_rms alto (pior {worst_div:.3e}). Ação: reduzir dt (dt_factor 0.8), aumentar pcg_max_iter (800) e/ou apertar pcg_tol (1e-6 → 5e-7).")

    # cv vs brinkman
    diffs = []
    for r in rows:
        if np.isfinite(r.get("Cl_brinkman", np.nan)) and np.isfinite(r.get("Cl_cv", np.nan)):
            diffs.append(abs(r["Cl_brinkman"] - r["Cl_cv"]))
    if diffs and float(np.mean(diffs)) > 0.25:
        rec.append("- CV e Brinkman discordam muito. Ação: ampliar cv_box (mais longe do aerofólio) e medir forças após assentamento (mais steps ou early-stop mais rígido).")

    # velocidade/vorticidade
    vmax = max((r["speed_max"] for r in rows if np.isfinite(r["speed_max"])), default=float("nan"))
    wmax = max((r["w_max"] for r in rows if np.isfinite(r["w_max"])), default=float("nan"))
    if np.isfinite(vmax) and vmax > 6.0:
        rec.append(f"- speed_max muito alto ({vmax:.2f}). Ação: dt menor e/ou domínio y maior (y_min/y_max) para reduzir recirculação artificial.")
    if np.isfinite(wmax) and wmax > 120.0:
        rec.append(f"- w_max muito alto ({wmax:.2f}). Ação: refinar um pouco malha (nx+20%) OU usar difusão implícita (futuro) para manter dt sem matar estabilidade.")

    if not rec:
        rec.append("- Nada gritante detectado. Próximo salto de qualidade: difusão implícita (Crank–Nicolson) + validação com XFOIL/UIUC.")

    lines.extend([r + "\n" for r in rec])

    # dump
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main():
    cfg = BaseConfig()

    root = ensure_dir(os.path.join("out", now_stamp()))
    ensure_dir(os.path.join(root, "states"))

    # salva config
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # domínio absoluto
    dom = {
        "x_min": cfg.x_min * cfg.c,
        "x_max": cfg.x_max * cfg.c,
        "y_min": cfg.y_min * cfg.c,
        "y_max": cfg.y_max * cfg.c,
    }
    geo = {"m": cfg.naca_m, "p": cfg.naca_p, "t": cfg.naca_t, "npts": cfg.naca_pts}
    pre = {"supersample": cfg.supersample, "sdf_smooth": cfg.sdf_smooth, "eps_cells": cfg.eps_cells}

    chi_cache = {}
    all_rows = []

    # ----------------------------
    # Experimentos (overnight, mas sem explodir PC)
    # ----------------------------
    # 1) Sweep alpha em Re=2000 (base)
    alphas = [-2, 0, 2, 4, 6, 8, 10, 12]
    base_grid = (320, 160)

    # 2) Sensibilidade dt em alpha=4
    dt_factors = [0.80, 1.00, 1.20]

    # 3) Sensibilidade grid em alpha=4 (custo controlado)
    grids = [(256, 128), (320, 160), (384, 192)]

    # 4) Checagem Re mais alto (pontos-chave)
    Re_checks = [(5000.0, [0, 4, 10])]

    # Pressão
    # Use MG apenas se você corrigiu o NaN do prolongamento; senão PCG.
    pressure_solver = os.environ.get("HARD_PRESSURE", cfg.pressure_solver)  # "pcg" ou "mg"
    if pressure_solver not in ("pcg", "mg"):
        pressure_solver = "pcg"

    # header CSV
    csv_path = os.path.join(root, "results.csv")
    fieldnames = [
        "case_id", "tag",
        "nx", "ny", "Re", "alpha_deg", "dt", "pressure_solver",
        "steps_ran", "stopped_early",
        "Cl", "Cd", "Cl_brinkman", "Cd_brinkman", "Cl_cv", "Cd_cv",
        "div_rms_post",
        "speed_mean", "speed_max", "ke_mean",
        "w_rms", "w_max", "enstrophy",
        "runtime_s", "s_per_step",
        "warm_state_loaded", "warm_state_saved",
    ]
    write_csv_header(csv_path, fieldnames)

    # ----------------------------
    # Runner sequencial (robusto)
    # ----------------------------
    def do_case(case):
        nx, ny = case["nx"], case["ny"]
        chi_pack = build_chi_cached(chi_cache, nx, ny, dom, geo, pre)

        warm = case.get("warm_state_path", None)
        try:
            row = run_one_case(case, cfg, chi_pack, root, warm_state_path=warm)
            append_csv_row(csv_path, fieldnames, row)
            all_rows.append(row)
            print(f"[OK] {row['case_id']} | Cl={row['Cl']:.3f} Cd={row['Cd']:.3f} | div={row['div_rms_post']:.3e} | {row['runtime_s']:.1f}s")
        except Exception as e:
            tb = traceback.format_exc()
            bad = {k: case.get(k, "") for k in ["case_id", "tag", "nx", "ny", "Re", "alpha_deg"]}
            bad.update({"error": str(e)})
            with open(os.path.join(root, "errors.log"), "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(json.dumps(bad, ensure_ascii=False) + "\n")
                f.write(tb + "\n")
            print(f"[FAIL] {case.get('case_id','?')}  (ver out/.../errors.log)")

    # 1) Sweep alpha (warm-start em cadeia)
    prev_state = None
    for a in alphas:
        case_id = f"sweep_Re2000_a{a:+03d}_{base_grid[0]}x{base_grid[1]}_{pressure_solver}"
        warm_path = prev_state
        do_case({
            "case_id": case_id,
            "tag": "sweep_alpha_Re2000",
            "nx": base_grid[0],
            "ny": base_grid[1],
            "Re": cfg.Re_baseline,
            "alpha_deg": float(a),
            "dom": dom,
            "pressure_solver": pressure_solver,
            "dt_factor": 1.00,
            "warm_state_path": warm_path,
        })
        # encadeia warm-start (estado do caso recém salvo)
        prev_state = os.path.join(root, "states", f"{case_id}.npz")

    # 2) dt sensitivity (alpha=4)
    for fdt in dt_factors:
        case_id = f"dt_sens_Re2000_a+04_f{fdt:.2f}_{base_grid[0]}x{base_grid[1]}_{pressure_solver}"
        do_case({
            "case_id": case_id,
            "tag": "dt_sensitivity",
            "nx": base_grid[0],
            "ny": base_grid[1],
            "Re": cfg.Re_baseline,
            "alpha_deg": 4.0,
            "dom": dom,
            "pressure_solver": pressure_solver,
            "dt_factor": float(fdt),
            "warm_state_path": None,
        })

    # 3) grid sensitivity (alpha=4)
    for (nx, ny) in grids:
        case_id = f"grid_sens_Re2000_a+04_{nx}x{ny}_{pressure_solver}"
        do_case({
            "case_id": case_id,
            "tag": "grid_sensitivity",
            "nx": int(nx),
            "ny": int(ny),
            "Re": cfg.Re_baseline,
            "alpha_deg": 4.0,
            "dom": dom,
            "pressure_solver": pressure_solver,
            "dt_factor": 1.00,
            "warm_state_path": None,
        })

    # 4) Re checks (pontos)
    for (Re, alist) in Re_checks:
        prev_state = None
        for a in alist:
            case_id = f"Re{int(Re)}_a{a:+03d}_{base_grid[0]}x{base_grid[1]}_{pressure_solver}"
            do_case({
                "case_id": case_id,
                "tag": "Re_checks",
                "nx": base_grid[0],
                "ny": base_grid[1],
                "Re": float(Re),
                "alpha_deg": float(a),
                "dom": dom,
                "pressure_solver": pressure_solver,
                "dt_factor": 0.90,          # um pouco mais conservador
                "warm_state_path": prev_state,
            })
            prev_state = os.path.join(root, "states", f"{case_id}.npz")

    # ----------------------------
    # Plots + report
    # ----------------------------
    plot_sweep(root, all_rows, "sweep_alpha_Re2000", "Re=2000")
    plot_sensitivity(
        root, all_rows, "dt_sensitivity",
        xkey="dt", ykeys=["Cl", "Cd", "div_rms_post"],
        fname="dt_sensitivity.png",
        title="Sensibilidade de dt (alpha=4, Re=2000)"
    )
    # grid plot (usa “nx” como x)
    plot_sensitivity(
        root, all_rows, "grid_sensitivity",
        xkey="nx", ykeys=["Cl", "Cd", "div_rms_post", "runtime_s"],
        fname="grid_sensitivity.png",
        title="Sensibilidade de malha (alpha=4, Re=2000)"
    )

    make_report(root, cfg, all_rows)

    print("\n=== FINAL ===")
    print(f"Saídas em: {root}")
    print("- results.csv, report.md, plots *.png, states/*.npz")


if __name__ == "__main__":
    main()
