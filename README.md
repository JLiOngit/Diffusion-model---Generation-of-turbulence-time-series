Diffusion Models for Synthetic Lagrangian Turbulence — Time Series Generation

This repository contains an academic project exploring the generation of synthetic Lagrangian turbulence time series using denoising diffusion probabilistic models (DDPM).
The work is inspired by the paper “Synthetic Lagrangian turbulence by generative diffusion models”, which demonstrated that diffusion models can accurately reproduce statistical and dynamical properties of turbulent flows.

The goal of this project is to reproduce key aspects of that methodology, implement a diffusion model for one-dimensional turbulent velocity signals, and evaluate the realism of the generated time series through classical turbulence statistics.

📌 Motivation

Turbulence is a complex multiscale phenomenon governed by nonlinear interactions across a wide range of scales.
Traditional numerical simulations (DNS/LES) are computationally expensive, while analytical modelling often fails to capture intermittent structures and long-time correlations.

Recent advances in generative AI — notably diffusion models — offer a promising alternative by learning the statistical structure of turbulence directly from data.

This project aims to:

explore diffusion models as a tool for data-driven turbulence synthesis,

compare generated trajectories to real Lagrangian datasets,

understand how well diffusion models recover classical turbulence markers (PDF shapes, structure functions, power spectra, autocorrelation).

📚 Summary of the Reference Paper

The reference work demonstrates that diffusion models:

learn the non-Gaussian, intermittent statistics of turbulent velocity increments,

generate synthetic trajectories consistent with Kolmogorov-type scaling laws,

reproduce both short-term dynamics and long-term temporal correlations,

outperform GANs and VAEs in turbulent data synthesis.

This project partially reproduces these ideas in a simplified form.

🧠 How Diffusion Models Work (ASCII diagram)
                FORWARD PROCESS (ADD NOISE)
            ----------------------------------
 real data x_0 → x_1 → x_2 → … → x_T (pure noise)

 each step: x_t = sqrt(1 - β_t) * x_(t-1) + sqrt(β_t) * ε

            REVERSE PROCESS (DENoise)
            ----------------------------------
 noise z_T → z_(T-1) → … → z_0 ≈ synthetic data

 model learns: ε_θ(x_t, t) ≈ real noise ε


The model learns to progressively remove noise and reconstruct statistically consistent turbulent time series.

📂 Repository Structure
├── data/                    # Sample datasets
├── models/                  # Diffusion model architecture & utilities
├── results/                 # Generated figures and outputs
├── demo.ipynb               # Main demonstration notebook
├── README.md                # Project documentation
└── requirements.txt

🚀 Running the Notebook

To reproduce the results:

pip install -r requirements.txt
jupyter notebook demo.ipynb


The notebook guides you through:

Data loading

Model configuration

Forward diffusion process

Reverse denoising sampling

Statistical validation of generated turbulence

📊 Analysis of Results (Detailed Interpretation)

Below is a full analysis of each major plot and evaluation step performed in demo.ipynb.

1. Training Loss Curve

Observation:
The training loss decreases smoothly and stabilizes after several epochs, with no sign of divergence.

Interpretation:

The model successfully learns the noise prediction task.

No evidence of mode collapse (unlike GANs).

The plateau indicates the model converged to a stable noise estimator.

2. Real vs Generated Time Series

Observation:
Generated signals visually resemble turbulent velocity fluctuations:

intermittent bursts

sharp gradients

irregular amplitude variations

non-stationary local behaviour

Interpretation:
The diffusion model correctly reproduces qualitative turbulent features.
However, small deviations in high-frequency content suggest limited capture of the smallest scales, likely due to training data size or network depth.

3. Probability Density Function (PDF)

Observation:
The PDF of generated velocities matches the real data closely—particularly the heavy tails.

Interpretation:

Diffusion models excel at capturing non-Gaussian statistics, a strong point of the DDPM approach.

Tail behaviour is reproduced, indicating successful modelling of intermittency.

Minor differences may appear near the distribution core (|v| < 1σ), suggesting smoothing by the model.

4. Power Spectral Density (PSD)

Observation:
The PSD slope of the generated signal approximates the real signal over a broad frequency range.

Interpretation:

The model reproduces the energy cascade signature, though the inertial range may be narrower.

Deviations at high frequencies indicate underfitting of small-scale turbulent structures.

This mirrors limitations observed in earlier generative models.

5. Autocorrelation Function

Observation:
The autocorrelation of synthetic data decays similarly to real turbulence but with slightly weaker long-time memory.

Interpretation:

Diffusion models recover short-time temporal dynamics well.

Long-time correlations (large-scale eddies) are partially captured but smoothed.

This is likely due to the model being 1D and without explicit temporal conditioning.

6. Increment Statistics / Structure Functions

Observation:
Real and generated increment PDFs align closely for small lags, diverging modestly at larger scales.

Interpretation:

The model reproduces intermittency at short timescales.

Larger-scale increments depend on long-range temporal relationships that the model only approximates.

✔️ Overall Model Quality

Strengths:

Good reproduction of heavy-tailed PDFs (intermittency).

Realistic time-series appearance.

Stable training and sampling.

Consistent PSD slopes and short-time structure functions.

Limitations:

Long-term correlations partially underestimated.

Small-scale high-frequency fluctuations smoothed out.

Scaling laws only approximated, not perfectly matched.

🔮 Future Work

To improve the fidelity of synthetic turbulence:

Conditioned diffusion models

introduce temporal conditioning

use transformer-based noise predictors

Multidimensional turbulence

extend from 1D velocity to 3D Lagrangian trajectories

Physics-informed diffusion

enforce scaling laws during training

embed Kolmogorov constraints into the loss

Longer sequences

train on larger datasets to enhance long-time statistics

Score-based SDE models

continuous-time formulation for improved small-scale accuracy
