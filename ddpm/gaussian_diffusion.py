import numpy as np
import torch as th
import math


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.99999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.00001
        beta_end = scale * 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name.startswith("power"):
        power = int(schedule_name[5:])
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - t**power,
        )
    elif schedule_name.startswith("exp"):
        t0 = float(schedule_name[3:])
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 2 - math.exp((t0 + math.log(2)) * t - t0),
        )
    elif schedule_name.startswith("tanh"):
        t0, t1 = schedule_name.split(",")
        t0, t1 = float(t0[4:]), float(t1)
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: -math.tanh((t0 + t1) * t - t0) + math.tanh(t1),
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    """
    
    def __init__(self,
                 betas,
                 device):
        self.device = device
        self.betas = th.tensor(betas, dtype=th.float32, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0], device=device)])
        self.num_diffusion_steps = len(betas)

    def forward(self, V0, n_batch, noise=None):
        """
        Add noise to the initial trajectories V0
        """
        V0 = V0.to(self.device)
        if noise is None:
            noise = th.randn_like(V0, device=self.device)
        else:
            noise = noise.to(self.device)
        alphas_cumprod = self.alphas_cumprod[n_batch.cpu().numpy()].to(device=self.device)
        while len(alphas_cumprod.shape) < len(V0.shape):
            alphas_cumprod = alphas_cumprod.unsqueeze(-1)
        Vn = th.sqrt(alphas_cumprod) * V0 + th.sqrt(1 - alphas_cumprod) * noise
        return Vn
    
    