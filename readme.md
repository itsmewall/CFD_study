# Navier-Stokes IBM Solver

Um solver de Dinâmica dos Fluidos Computacional (CFD) desenvolvido em Python para simular escoamentos incompressíveis 2D ao redor de perfis aerodinâmicos.

O projeto utiliza o **Método dos Limites Imersos (Immersed Boundary Method - IBM)** com penalização de Brinkman e é acelerado via **Numba** para alta performance, permitindo simulações rápidas em CPUs convencionais.

## 🚀 Destaques Técnicos

* **Arquitetura:** Malha desencontrada (MAC Grid) com Método de Projeção (Passo fracionado).
* **Advecção:** Esquema Semi-Lagrangeano com integração RK2 (incondicionalmente estável).
* **Solver de Pressão:** Gradiente Conjugado Pré-condicionado (PCG) para a equação de Poisson.
* **Geometria:** Geração de malha com Signed Distance Field (SDF) suavizado para reduzir efeitos de serrilhado na fronteira imersa ($\chi$ field).
* **Cálculo de Forças:** Implementação híbrida comparando integração direta (Brinkman) e Balanço de Momento em Volume de Controle (CV).
* **Performance:** Loops críticos otimizados e paralelizados com `@njit(parallel=True)` do Numba.

## 📦 Dependências

O projeto requer Python 3.11+ e as seguintes bibliotecas:

```bash
pip install numpy matplotlib numba

```

*(Opcional: `ffmpeg` se desejar salvar as animações em vídeo).*

## 📂 Estrutura do Projeto

* `mac_solver.py`: O "motor" da simulação. Contém o solver Navier-Stokes, rotinas de PCG, advecção e cálculo de forças.
* `geometry.py`: Gerador de coordenadas NACA 4 dígitos e criador da matriz de máscara sólida (\chi) com super-sampling.
* `run.py`: Script principal para rodar uma simulação única (single shot) com visualização em tempo real e animação de vorticidade.
* `sweep_alpha.py`: Script para gerar a polar de arrasto (C_L e C_D vs \alpha). Utiliza *warm-start* e *early-stopping* para eficiência.
* `compare_forces.py`: Estudo comparativo entre métodos de cálculo de força (Brinkman vs. Volume de Controle).

## 🛠️ Como Usar

### 1. Simulação Única (Visualização)

Para rodar uma simulação com visualização da vorticidade e partículas passivas (tracers):

```bash
python run.py

```

*Configuração padrão:* Re=2000, AoA=4°, Malha 320x160.
*Saída:* Plota o campo de velocidade final e gera uma animação da esteira de vórtices.

### 2. Gerar Polar Aerodinâmica (Sweep)

Para calcular C_L e C_D em vários ângulos de ataque (ex: -4° a +12°):

```bash
python sweep_alpha.py

```

*Saída:* Gera arquivos `.csv` e `.npz` com os dados e plota os gráficos de sustentação e arrasto. O solver usa o estado final do ângulo anterior como condição inicial do próximo para acelerar a convergência.

## 📊 Física e Métodos Numéricos

### Equações Governantes

O solver resolve as equações de Navier-Stokes incompressíveis com um termo de força de penalização:

$$ \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} - \frac{\chi}{\eta}\mathbf{u} $$

Onde \chi é a função de máscara (0 no fluido, 1 no sólido) e \eta é o parâmetro de permeabilidade (Brinkman).

### Cálculo de Forças

Devido às oscilações inerentes à integração da força de Brinkman na fronteira difusa, este solver implementa um método robusto de **Volume de Controle (CV)**, integrando o fluxo de momento através de uma caixa retangular ao redor do perfil para obter coeficientes C_L e C_D precisos.

---

**Autor:** Wallace de Oliveira Ferreira

```

