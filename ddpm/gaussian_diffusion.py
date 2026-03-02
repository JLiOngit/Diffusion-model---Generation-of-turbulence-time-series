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
        alphas_cumprod = self.alphas_cumprod[n_batch.cpu().numpy()]
        while len(alphas_cumprod.shape) < len(V0.shape):
            alphas_cumprod = alphas_cumprod.unsqueeze(-1)
        Vn = th.sqrt(alphas_cumprod) * V0 + th.sqrt(1 - alphas_cumprod) * noise
        return Vn

    def get_posterior_variance(self, n_batch):
        """
        Calculate the posterior variance of the trajectory Vn-1
        """
        alpha_cumprod_n = self.alphas_cumprod[n_batch.cpu().numpy()]
        alpha_cumprod_nprev = self.alphas_cumprod_prev[n_batch.cpu().numpy()]
        beta_n_coeff = (1 - alpha_cumprod_nprev) / (1 - alpha_cumprod_n)
        beta_n = 1 - alpha_cumprod_n / alpha_cumprod_nprev
        variance_nprev = th.clamp(beta_n_coeff * beta_n, min=1e-20)
        return variance_nprev

    def get_posterior_mean(self, n_batch, Vn, V0):
        """
        Calculate the posterior mean of the trajectory Vn-1
        """
        V0 = V0.to(self.device)
        Vn = Vn.to(self.device)
        alpha_cumprod_n = self.alphas_cumprod[n_batch.cpu().numpy()]
        alpha_cumprod_nprev = self.alphas_cumprod_prev[n_batch.cpu().numpy()]
        alpha_n = alpha_cumprod_n / alpha_cumprod_nprev
        Vn_coeff = th.sqrt(alpha_n) * (1 - alpha_cumprod_nprev) / (1 - alpha_cumprod_n)
        beta_n = self.betas[n_batch]
        V0_coeff = th.sqrt(alpha_cumprod_nprev) * beta_n / (1 - alpha_cumprod_n)
        while len(Vn_coeff.shape) < len(Vn.shape):
            Vn_coeff = Vn_coeff.unsqueeze(-1)
            V0_coeff = V0_coeff.unsqueeze(-1)
        mean_nprev = Vn_coeff * Vn + V0_coeff * V0
        return mean_nprev

    def get_Vnprev(self, n_batch, Vn, V0, noise=None):
        """
        Get a sample of the distribution p(Vn-1|Vn,V0)
        """
        V0 = V0.to(self.device)
        Vn = Vn.to(self.device)
        mean_nprev = self.get_posterior_mean(n_batch, Vn, V0)
        variance_nprev = self.get_posterior_variance(n_batch)
        while len(variance_nprev.shape) < len(mean_nprev.shape):
            variance_nprev = variance_nprev.unsqueeze(-1)
        if noise is None:
            noise = th.randn_like(V0, device=self.device)
        else:
            noise = noise.to(self.device)
        return mean_nprev + th.sqrt(variance_nprev) * noise
    
    def get_V0_from_noise(self, Vn, n_batch, noise):
        """
        Predict the initial trajectory given the noise
        """
        Vn, noise = Vn.to(self.device), noise.to(self.device)
        alpha_cumprod_n = self.alphas_cumprod[n_batch.cpu().numpy()]
        while len(alpha_cumprod_n.shape) < len(Vn.shape):
            alpha_cumprod_n = alpha_cumprod_n.unsqueeze(-1)
        V0 = (Vn - th.sqrt(1-alpha_cumprod_n) * noise) / th.sqrt(alpha_cumprod_n)
        return V0

    def training_losses(self, model, V0, n_batch, noise=None, loss_type='mse'):
        """
        Compute training losses for a single diffusion step n.
        """
        V0 = V0.to(self.device)
        n_batch = n_batch.to(self.device)
        if noise is None:
            noise = th.randn_like(V0, device=self.device)
        else:
            noise = noise.to(self.device)
        Vn = self.forward(V0, n_batch, noise)
        terms = {}
        predicted_noise = model(Vn, n_batch)
        if loss_type == 'mse':
            mse_loss = ((predicted_noise - noise)**2).mean(dim=list(range(1, len(noise.shape))))
            terms["mse"] = mse_loss
        elif loss_type == 'kl':
            noise_mean = noise.mean(dim=(1, 2), keepdim=True)
            noise_var = noise.var(dim=(1, 2), keepdim=True)
            noise_logvar = th.log(noise_var + 1e-30)
            pred_mean = predicted_noise.mean(dim=(1, 2), keepdim=True)
            pred_var = predicted_noise.var(dim=(1, 2), keepdim=True)
            pred_logvar = th.log(pred_var + 1e-30)
            kl = self.loss_kl(mean1=noise_mean,
                        logvar1=noise_logvar,
                        mean2=pred_mean,
                        logvar2=pred_logvar)
            kl_loss = kl.mean()
            terms["kl"] = kl_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return terms

    def loss_kl(mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two Gaussians: N(mean1, exp(logvar1)) || N(mean2, exp(logvar2))
        """
        while len(logvar1.shape) < len(mean1.shape):
            logvar1 = logvar1.unsqueeze(-1)
        while len(logvar2.shape) < len(mean2.shape):
            logvar2 = logvar2.unsqueeze(-1)
        kl = 0.5 * (1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * th.exp(-logvar2))
        return kl
    
    