# Project Overview & Academic Background

This repository provides the implemented code for the Recent Advances in Machine Learning course project, part of the MEng in Computer Science program at IMT Atlantique.

This work is based on the paper Li, T., Biferale, L., Bonaccorso, F., Scarpolini, M. A., & Buzzicotti, M. (2024). [*"Synthetic Lagrangian turbulence by generative diffusion models"*.](https://doi.org/10.1038/s42256-024-00810-0).

The goal of this project is to reproduce and analyze how Denoising Diffusion Probabilistic Models (DDPM) can serve as a computationally efficient alternative to costly Direct Numerical Simulations (DNS) for generating stochastic processes that preserve the key multi-scale statistics and intermittent behavior of turbulent flows.

---

# Dataset

The research paper uses turbulent Lagrangian trajectories extracted from a high-resolution Direct Numerical Simulation (DNS) of the three-dimensional incompressible Navier–Stokes equations in a cubic, periodic domain with homogeneous isotropic forcing, once statistical stationarity had been reached.

Each trajectory was sampled at a temporal resolution of $\Delta t \simeq 0.1\,\tau_\eta
$ where $\tau_\eta$ is the Kolmogorov time scale. This corresponds to **2,000 time steps per trajectory**, corresponding to $T \simeq 200\,\tau_\eta$ of Lagrangian evolution.

A subset of **768 one-dimensional (1D) turbulent velocity trajectories** was made publicly available and used in our implementation. Each trajectory represents the time evolution of a single velocity component of a tracer particle in the turbulent flow. 

![''](plots/time_series.png)

---


# Turbulence properties

This section summarizes the main statistical quantities used to assess whether a generative model correctly reproduces the physics of Lagrangian turbulence.

## 1. The Probability density function of increments

The Probability Density Function (PDF) of velocity increments  $\delta_\tau u = u(+\tau) - u(t)$ measures the likelihood of observing a given change in velocity over a time lag $\tau$.

![''](plots/pdf_increments.png)

The shape of the PDF changes with the scale $\tau$:
- **Large scales $\tau$**: the PDF is close to Gaussian so velocity variations are relatively smooth and predictable.
- **Small scales $\tau$**: the PDF develops *fat tails* indicating that extreme events (strong, sudden accelerations) occur far more frequently than in a purely random Gaussian process. This phenomenon is known as **intermittency**

If the model reproduces these fat tails, it successfully captures the **fine-scale intermittent** structures of turbulence.

## 2. Lagrangian Structure Functions

The structure function of order p defined as $S_p(\tau) = \langle [\delta_\tau u]^p \rangle$ quantifies the magnitude of velocity fluctuations across scales, capturing the **energy distribution** and **multi-scale correlations** of turbulence. $\tau$. 
For $p=2$, $S_2(\tau)$ represents the energy associated with fluctuations at scale $\tau$.

In fully developed turbulence, the structure functions follow scaling laws in the inertial range: $S_p(\tau) \propto \tau^{\zeta_p}$ where $\zeta_p$ are scaling exponents. Deviations from linear scaling indicate **intermittency**.


## 3. Generalized Flatness

The generalized flatness of order \(p\) defined as $F_p(\tau) = \frac{S_p(\tau)}{[S_ (\tau)]^{p/2}}$ measures the **degree of intermittency**: how much the distribution of increments deviates from Gaussian behavior. Large $F_p(\tau)$ at small $\tau$ indicates frequent **extreme events** and strong small-scale turbulent bursts.


![''](plots/structure_flatness.png)

# The DDPM Mathematical Framework

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. They work by:
1. **Forward process**: Gradually adding Gaussian noise to data until it becomes pure noise
2. **Reverse process**: Learning to denoise, step by step, from pure noise back to data

## 1. Forward Diffusion Process

The forward process gradually corrupts data $x_0$ by adding Gaussian noise over $T$ timesteps.

### Markov Chain Definition

$$q(x_{1:T} | x_0) := \prod_{t=1}^{T} q(x_t | x_{t-1})$$

where each step adds noise:

$$q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

**With key parameters:**
- $\beta_t$: Variance schedule controlling noise added at step $t$
- $T$: Total diffusion steps

We defined and displayed different variance schedulers. 
We decided to use a linear scheduler and T = 800 diffusion steps, rather than the tanh6,1 scheduler used in the paper, after few experiences showing a lower loss from the model

![Different variance schedulers](plots/variance_scheduler.png)


This reflects how information gradually dissipates: the structured and intermittent bursts in the velocity signal are increasingly damaged during the process, until they eventually reach a state of maximum entropy, similar to pure white noise. 

In addition, we displayed the evolution of the time series and its statistical properties at different diffusion steps in order to visualise the forward process of the diffusion model.

![Forward diffusion process](plots/forward.png)

### 2. The Reverse Denoising Process ($p_\theta$)
The generative model learns to reverse the diffusion by approximating the conditional distribution $q(x_{t-1} | x_t)$. We use a learned model $p_\theta$:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

In our implementation, the 1D U-Net is trained to predict the noise $\epsilon_\theta$ added at step $t$. More precisely, the training objective (loss function) is to minimize the difference between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta$. This is equivalent to **denoising score matching**:

$$L_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]$$

![Training and validation losses](plots/losses.png)

Once the model is trained, we can generate new turbulent trajectories by **starting from pure Gaussian noise** and iteratively applying the reverse diffusion process. At each step t, assuming that the standard deviations of the reverse and forward processes are
identical ($\Sigma_\theta(x_t, t) = \beta_tI$), the model predicts the mean of the previous trajectory $x_{t-1}$ conditioned on the current noisy trajectory $x_t$ and the time step t. The update rule is given by:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \beta_t \frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \right)$$

The actual update step adds stochasticity:

$$x_{t-1} = \mu_\theta(x_t, t) + \beta_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

Here, we show the backward diffusion process at different steps, illustrating how the model progressively removes noise to generate realistic turbulent trajectories.

![''](plots/backward.png)

We can observe that, although the model is able to generate trajectories from pure Gaussian noise, some residual noise remains in the final samples. Further modifications and experiments could be explored to improve the model's performance and reduce these remaining artifacts.

Finally, a total of 250 trajectories were generated. These trajectories are used to compute the previously introduced turbulence metrics and to evaluate how well the model reproduces the key characteristics of the turbulent flow.

![''](plots/generated_pdf_increments.png)

![''](plots/generated_structure_flatness.png)


