# **Project Overview**

This repository provides the implemented code for the **Recent Advances in Machine Learning course project**, as part of the **MEng in Computer Science program at IMT Atlantique**.

The goal of this project is to explore and analyze how **Denoising Diffusion Probabilistic Models (DDPM)** can be used as an efficient alternative to costly Direct Numerical Simulations (DNS) for generating stochastic processes that preserve the **key multi-scale statistics** and **intermittent behavior** of turbulent flows.

# **Installation**

```bash
# Clone the repository
git clone https://github.com/JLiOngit/Diffusion-model---Generation-of-turbulence-time-series.git
cd Diffusion-model---Generation-of-turbulence-time-series

# Install dependencies
pip install -r requirements.txt
```

# **Running**

The project enables to :
- Train the 1D U-Net diffusion model (`--action = only_training`)
- Generate synthetic turbulent trajectories (`--action = only_sampling`), once you have a model which has been trained and stored in `trained_model/`
- Perform both sequentially (`--action = training_and_sampling`).

Configuration parameters are defined in `ddpm/main.py` (class `Config`). Training and sampling parameters (`BATCH_SIZE`, `LEARNING_RATE`, `NUM_SAMPLES`, etc.) can also be overridden via CLI arguments.

```bash
python -m ddpm.main [OPTIONS]
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--action` | `str` |`training_and_sampling` | `only_training`, `only_sampling`, or `training_and_sampling` |
| `--num_diffusion` | `int` | `800` | Number of diffusion steps |
| `--schedule_name` | `str` | `linear` | Beta schedule name |
| `--batch_size` | `int` | `32` | Batch size |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--weight_decay` | `float` | `1e-2` | Weight decay |
| `--ema_rate` | `float` | `0.97` | EMA rate |
| `--loss_type` | `str` | `mse` | Loss type (`mse` or `kl`) |
| `--num_train` | `int` | `500` | Number of training steps |
| `--num_validation` | `int` | `25` | Number of validation steps |
| `--num_samples` | `int` | `500` | Number of samples to generate |

# **Reference**

[1] Li, T., Biferale, L., Bonaccorso, F., Scarpolini, M. A., & Buzzicotti, M. (2024). Synthetic Lagrangian turbulence by generative diffusion models.


