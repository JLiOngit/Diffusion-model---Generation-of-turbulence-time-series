import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from .stats import *


def plot_time_series(sample):
    """
    Plot one sample time serie.
    """
    if isinstance(sample, th.Tensor):
        sample = sample.detach().cpu().numpy()
    time = np.arange(sample.shape[1])
    figure, ax = plt.subplots(figsize=(15,5))
    figure.suptitle(f'A sample of {sample.shape[0]}D turbulent velocity trajectory', fontsize=18)
    for i in range(sample.shape[0]):
        ax.plot(time, sample[i,:], label=f"V{i+1}")
    ax.set_xlabel(r'Number of timesteps (t = 0.1$\mathcal{\tau}_n$)', fontsize=15, labelpad=8)
    ax.set_ylabel('Velocity', fontsize=15, labelpad=8)
    ax.tick_params('x', labelsize=10)
    ax.tick_params('y', labelsize=10)
    ax.legend(fontsize=15)
    plt.show()


def plot_pdf_increments(velocities, tau_values):
    """
    Plot the probability density function (PDF) of velocity increments.
    """
    fig, ax = plt.subplots(figsize=(15,5))
    fig.suptitle('Standardized PDFs of velocity increments for different lag τ', fontsize=18)
    for tau in tau_values:
        increments = velocity_increments(velocities, tau)
        kde = gaussian_kde(increments)
        std_increments = increments / increments.std()
        x_values = np.linspace(std_increments.min(), std_increments.max(), 1000)
        y_values = kde(x_values)
        ax.plot(x_values, y_values, label=r"$\mathcal{\tau}$ =" + f'{tau//10}' + r"$\mathcal{\tau}_n$")
    ax.set_xlabel('δτVi/σ(δτVi)', fontsize=15, labelpad=8)
    ax.set_ylabel('PDF(δτVi)', fontsize=15, labelpad=8)
    ax.tick_params('x', labelsize=10)
    ax.tick_params('y', labelsize=10)
    ax.legend(fontsize=15)
    plt.show()


def plot_structure_flatness(velocities, p_values, tau_values):
    """
    Lagrangian structure function and flatness for different orders p
    """
    fig, axes = plt.subplots(ncols=2, figsize=(15,6))
    axes = axes.flatten()
    for p in p_values:
        structure_p_values = structure_p(velocities, p, tau_values)
        flatness_p_values = flatness_p(velocities, p, tau_values)
        sns.lineplot(x=tau_values, y=structure_p_values, ax=axes[0], label=f'p = {p}', marker="o")
        sns.lineplot(x=tau_values, y=flatness_p_values, ax=axes[1], label=f'p = {p}', marker="o")
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('τ')
    axes[0].set_ylabel('S(p)τ')
    axes[0].set_title('log–log plot of Lagrangian structure functions')
    axes[0].legend()
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('τ')
    axes[1].set_ylabel('F(p)τ')
    axes[1].set_title(' log–log plot of the generalized flatness')
    axes[1].legend()
    plt.tight_layout()
    plt.show()