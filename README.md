# Synthetic Lagrangian Turbulence Generation via Denoising Diffusion Probabilistic Models (DDPM)

## 📖 Project Overview & Academic Background

This project implements a generative framework for synthesizing **Lagrangian turbulence time series**. It is an academic reproduction and exploration based on the state-of-the-art research published in *Nature Machine Intelligence*:

> **Primary Reference:** Li, T., Biferale, L., Bonaccorso, F., Scarpolini, M. A., & Buzzicotti, M. (2024). *"Synthetic Lagrangian turbulence by generative diffusion models"*. [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00810-0).

The objective is to demonstrate that **Generative AI** can circumvent the massive computational cost of Direct Numerical Simulations (DNS) while faithfully preserving the multi-scale statistics and intermittent nature of turbulent flows.

---

## 🔬 Ground-Truth Analysis & Baseline Metrics

Before training, the `demo.ipynb` notebook performs a detailed diagnostic of the reference DNS data:
- **Trajectory Analysis:** Inspection of the velocity $u(t)$ to observe chaotic fluctuations and temporal persistence.
- **Statistical Signature:** Establishing the reference for the Probability Density Functions (PDF) and Power Spectral Density (PSD) that the model must replicate.

---

## 🧠 Deep Dive: The DDPM Mathematical Framework

The core implementation follows the **Denoising Diffusion Probabilistic Models (DDPM)** framework defined by *Ho et al. (2020)*. The process is divided into two distinct Markov chains.

### 1. The Forward Diffusion Process ($q$)
The forward process is a fixed Markov chain that gradually adds Gaussian noise to the initial turbulent signal $x_0$ over $T$ steps, according to a variance schedule $\beta_1, \dots, \beta_T$:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

A fundamental property of this process is that it allows sampling $x_t$ at any arbitrary timestep $t$ in closed form, using the notation $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

* **Physical Analogy:** This mirrors the dissipation of information where the structured, intermittent bursts of the velocity signal are progressively destroyed until they reach a state of maximum entropy (pure white noise).

### 2. The Reverse Denoising Process ($p_\theta$)
The generative model learns to reverse the diffusion by approximating the conditional distribution $q(x_{t-1} | x_t)$. We use a learned model $p_\theta$:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

In our implementation, the 1D U-Net is trained to predict the noise $\epsilon$ added at step $t$. The sampling (generation) follows a stochastic decay process:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t \mathbf{z}, \quad \text{where } \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

### 3. Training Objective (Loss Function)
The model is trained to minimize the difference between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta$. This is equivalent to **denoising score matching**:

$$L_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]$$

## 📊 Detailed Turbulence Analysis & Results

The validation compares the generated trajectories against the initial ground-truth trajectories across three fundamental pillars.

### 1. Intermittency & PDFs of Increments
Turbulence is characterized by **intermittency**, where small-scale structures are highly non-Gaussian.
- **Analysis:** PDFs of velocity increments $\delta_\tau u = u(t+\tau) - u(t)$. 
- **Observations:** For small $\tau$, the PDFs exhibit **heavy-tailed distributions** (fat tails).
- **Result:** Our DDPM successfully captures these fat tails, representing rare but intense acceleration events. This confirms the model's ability to reproduce **anomalous scaling**.



### 2. The Energy Cascade (Spectral Analysis)
Energy cascades from large scales down to small scales where it dissipates.
- **Analysis:** Power Spectral Density (PSD) calculation.
- **Result:** The synthetic signal follows the expected Lagrangian power-law $E(f) \propto f^{-2}$, confirming that the model respects the **Kolmogorov phenomenology**.

### 3. Temporal Coherence (Autocorrelation)
- **Analysis:** Comparison of the Autocorrelation Function (ACF) to evaluate temporal memory.
- **Result:** The model accurately reproduces the **integral time scale** $T_L$, ensuring the synthetic particles drift with the same temporal persistence as real ones.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- PyTorch
- NumPy / Matplotlib / Scipy
