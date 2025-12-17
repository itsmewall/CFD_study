# Relatório — hard_run

- Data: 2025-12-17 12:40:15

- Runs totais: 17 (ok: 17)

- Config base: steps=2800, out_every=250, eta=0.001, force_method=both


## Performance

- Tempo médio por caso: 2467.48s (min 261.92s, max 32479.10s)

- Tempo médio por step: 0.883000s/step


## Sweep alpha (Re=2000)

- Pontos: 8

- Inclinação Cl_alpha (faixa -2..6 deg): 4.344 por rad (R²=0.981)

- div_rms pós (médio): 1.689e-03 | (pior): 2.041e-03

- |ΔCl| médio (Brinkman vs CV): 0.578

- |ΔCd| médio (Brinkman vs CV): 0.186


## Diagnóstico e recomendações (automático)

- CV e Brinkman discordam muito. Ação: ampliar cv_box (mais longe do aerofólio) e medir forças após assentamento (mais steps ou early-stop mais rígido).
