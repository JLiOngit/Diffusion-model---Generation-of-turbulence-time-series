import math
import numpy as np
import torch as th
from .gaussian_diffusion import *


class Sample_loop:

    def __init__(self,
                 model,
                 betas,
                 noise,
                 device):
        self.betas = th.tensor(betas, dtype=th.float32, device=device)
        self.model = model.to(device)
        self.device = device
        self.noise = noise.to(device)
        self.model_diffusion = GaussianDiffusion(betas, device)
        self.num_diffusion_steps = len(betas)

    def get_model_mean_variance(self, n, Vn):
        """
        Predict the distribution p(x_t-1 | x_t) (mean and variance), as well as a prediction of the initial trajectories V0.
        """
        with th.no_grad():
            Vn = Vn.to(self.device, dtype=th.float32)
            n_batch = th.tensor([n] * Vn.shape[0], device=self.device)
            predicted_noise = self.model(Vn, n_batch)
            predicted_V0 = self.model_diffusion.get_V0_from_noise(Vn, n_batch, predicted_noise)
            predicted_mean = self.model_diffusion.get_posterior_mean(n_batch, Vn, predicted_V0)
            predicted_variance = self.model_diffusion.get_posterior_variance(n_batch)
            output = {
                'predicted mean': predicted_mean,
                'predicted variance': predicted_variance,
                'predicted V0': predicted_V0,
                'predicted noise': predicted_noise
            }
            return output
    
    def sample_Vnprev(self, n, Vn):
        """
        Sample Vn-1 from the distribution predicted by the model at the given diffusion step n.
        """
        with th.no_grad():
            Vn = Vn.to(self.device)
            output = self.get_model_mean_variance(n, Vn)
            nonzero_mask = (th.tensor(n != 0, device=Vn.device, dtype=Vn.dtype)).view(-1, *([1] * (len(Vn.shape) - 1)))
            noise = th.randn_like(Vn)
            sample = output['predicted mean'] + nonzero_mask * th.sqrt(output['predicted variance']) * noise
            samples_dict = {'sample':sample, 'predicted V0':output['predicted V0'], 'predicted mean':output['predicted mean'], 'predicted noise':output['predicted noise']}
            return samples_dict
    
    def run_loop(self):
        """
        Generate samples from the model and yield intermediate samples from each timestep of diffusion.
        """
        with th.no_grad():
            diffusion_steps = list(range(self.num_diffusion_steps))[::-1]
            Vn = self.noise
            for n in diffusion_steps:
                sample_dict = self.sample_Vnprev(n, Vn)
                Vn = sample_dict['sample']
                yield sample_dict