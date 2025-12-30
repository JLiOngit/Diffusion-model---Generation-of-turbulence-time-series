import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from .stats import *
from ddpm.gaussian_diffusion import *


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
    axes[0].set_xlabel('τ/τn')
    axes[0].set_ylabel('S(p)τ')
    axes[0].set_title('log–log plot of Lagrangian structure functions')
    axes[0].legend()
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('τ/τn')
    axes[1].set_ylabel('F(p)τ')
    axes[1].set_title(' log–log plot of the generalized flatness')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_forward(V0, num_diffusion_steps, schedule_name, diffusion_steps_ratio=[0.3,0.6,0.9,1]):
    """
    Plot the evolution graphes of forward process for different diffusion steps calculated with diffusion_steps_ratio:
        graph 1 : The diffusion step n and the corresponding alpha bar value
        graph 2 : Time-series of the noisy trajectory Vn
        graph 3 : Comparison of standardized PDF of δτVi between both Vn and V0 increments at τ=τn
        graph 4 : Comparison of structure function for p = 2 between both Vn and V0
        graph 5 : Comparison of structure function for p = 4 between both Vn and V0
    """
    # Define the grid of plots
    fig, axes = plt.subplots(nrows=len(diffusion_steps_ratio), ncols=5, figsize=(90,70))
    # Define the title of each column
    col_titles = ['Noise schedule', 'Time series', r'Velocities increments PDF ($\mathcal{\tau}$ = $\mathcal{\tau}_n$)', 'Structure function S(p)τ for p=2', 'Generalized flatness F(p)τ for p=4' ]
    for i, ax in enumerate(axes[0]):
        ax.set_title(col_titles[i], fontsize=60)

    betas = get_named_beta_schedule(schedule_name, num_diffusion_steps)
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    diffusion_model = GaussianDiffusion(betas, device)
    noise = th.randn_like(V0)

    # Loop over the different diffusion steps
    for (i, ratio) in enumerate(diffusion_steps_ratio):
        n  = int(min(num_diffusion_steps*ratio, num_diffusion_steps-1))
        n_batch = th.tensor([n] * V0.shape[0])
        alpha_cumprods = diffusion_model.alphas_cumprod.cpu().numpy()
        Vn = diffusion_model.forward(V0, n_batch, noise)
        # Retrieve a sample for plotting the time serie
        Vn_sample = Vn[0,0,:]
        # Plot the graph 1
        sns.lineplot(x = np.arange(num_diffusion_steps),
                    y = alpha_cumprods,
                    ax=axes[i,0],
                    linewidth=3)
        sns.scatterplot(x = [n],
                        y = [alpha_cumprods[n].item()],
                        marker = 'x',
                        color = 'red',
                        ax=axes[i,0],
                        label=f'n = {n}',
                        s=1500)
        axes[i,0].set_xlabel('Diffusion step', fontsize=40, labelpad=25)
        axes[i,0].set_ylabel(r'$\bar{\alpha}_n$', fontsize=40)
        axes[i,0].tick_params('x', labelsize=30)
        axes[i,0].tick_params('y', labelsize=30)
        axes[i,0].legend(fontsize=30)
        # Plot the graph 2
        sns.lineplot(x = np.arange(len(Vn_sample)),
                    y = Vn_sample.cpu().numpy(),
                    ax=axes[i,1],
                    linewidth=3,
                    label = r'$\bar{\alpha}_n$ = ' + f"{alpha_cumprods[n]:.2f}")
        axes[i,1].set_xlabel('Timestep', fontsize=40, labelpad=25)
        axes[i,1].set_ylabel('Velocity', fontsize=40, labelpad=25)
        axes[i,1].tick_params('x', labelsize=30)
        axes[i,1].tick_params('y', labelsize=30)
        axes[i,1].legend(fontsize=30)
        # Plot the graph 3
        # Calculate Vn and gaussian increments at τ = τn
        Vn_increments = velocity_increments(Vn, 10).cpu().numpy()
        V0_increments = velocity_increments(V0, 10).cpu().numpy()
        Vn_std_increments = Vn_increments / Vn_increments.std()
        x_values = np.linspace(Vn_std_increments.min(), Vn_std_increments.max(), 1000)
        # Calculate the kde of both increments and plot values of standardized increments
        Vn_kde = gaussian_kde(Vn_increments)
        Vn_values = Vn_kde(x_values)
        V0_kde = gaussian_kde(V0_increments)
        V0_values = V0_kde(x_values)
        sns.lineplot(x = x_values,
                    y = V0_values,
                    ax=axes[i,2],
                    linewidth=3,
                    color='red',
                    label='V0')
        sns.lineplot(x = x_values,
                y = Vn_values,
                ax=axes[i,2],
                linewidth=3,
                label=f'V{n}')
        axes[i,2].set_xlabel('δτVi/σ(δτVi)', fontsize=40, labelpad=25)
        axes[i,2].set_ylabel('PDF of δτVi', fontsize=40, labelpad=25)
        axes[i,2].tick_params('x', labelsize=30)
        axes[i,2].tick_params('y', labelsize=30)
        axes[i,2].legend(fontsize=30)
        # Plot the graph 4
        tau_values = [i * 10**j for j in range(3) for i in range(1,10)]
        structure_V0 = structure_p(V0, 2, tau_values)
        structure_Vn = structure_p(Vn, 2, tau_values)
        sns.lineplot(x=tau_values, y=structure_V0, ax=axes[i,3], label=f'V0', marker="o")
        sns.lineplot(x=tau_values, y=structure_Vn, ax=axes[i,3], label=f'V{n}', marker="o")
        axes[i,3].set_yscale('log')
        axes[i,3].set_xscale('log')
        axes[i,3].set_xlabel('τ/τn', fontsize=40, labelpad=25)
        axes[i,3].set_ylabel('S(p)τ', fontsize=40, labelpad=25)
        axes[i,3].tick_params('x', labelsize=30)
        axes[i,3].tick_params('y', labelsize=30)
        axes[i,3].legend(fontsize=30)
        # Plot the graph 5
        flatness_V0 = flatness_p(V0, 4, tau_values)
        flatness_Vn = flatness_p(Vn, 4, tau_values)
        sns.lineplot(x=tau_values, y=flatness_V0, ax=axes[i,4], label=f'V0', marker="o")
        sns.lineplot(x=tau_values, y=flatness_Vn, ax=axes[i,4], label=f'V{n}', marker="o")
        axes[i,4].set_yscale('log')
        axes[i,4].set_xscale('log')
        axes[i,4].set_xlabel('τ/τn', fontsize=40, labelpad=25)
        axes[i,4].set_ylabel('F(p)τ', fontsize=40, labelpad=25)
        axes[i,4].tick_params('x', labelsize=30)
        axes[i,4].tick_params('y', labelsize=30)
        axes[i,4].legend(fontsize=30)


def plot_backward(sample_outputs, parameter, model_diffusion, diffusion_steps_ratio=[0, 0.1, 0.2, 0.5, 0.7, 0.9]):
    """
    Plot the evolution of predicted previous parameter from the backward process for different diffusion steps calculated with diffusion_steps_ratio.
    The predicted parameters are :
        - predicted Vn-1 sample
        - predicted Vn-1 mean
        - predicted V0 calculated from the predicted noise
        - predicted noise added to Vn (except for V0)
    """
    N = len(sample_outputs['sample'])
    diffusion_steps_plot = [int(min(r * N, N - 1)) for r in diffusion_steps_ratio][::-1]
    fig, axes = plt.subplots(nrows=len(diffusion_steps_ratio), ncols=2, figsize=(25,30))
    col_titles = ['Diffusion step', f'Vn-1 {parameter} Time Series']
    for i, ax in enumerate(axes[0]):
        ax.set_title(col_titles[i], fontsize=20)
    for (j, step) in enumerate(diffusion_steps_plot):
        data = sample_outputs[parameter][N - 1 - step].detach().cpu().numpy().flatten()
        sns.lineplot(x=np.arange(N),
                    y=model_diffusion.alphas_cumprod.cpu().numpy(),
                    ax=axes[j, 0], linewidth=2)
        sns.scatterplot(x=[step],
                        y=[model_diffusion.alphas_cumprod[step].item()],
                        marker='x', color='red', s=200,
                        ax=axes[j, 0])
        axes[j, 0].set_xlabel('Step', fontsize=16)
        axes[j, 0].set_ylabel(r'$\bar{\alpha}_n$', fontsize=16)
        axes[j, 0].tick_params(labelsize=14)
        sns.lineplot(x = np.arange(len(data)),
                     y = data,
                     ax=axes[j,1],
                     linewidth=2)
        axes[j, 1].set_xlabel('Timestep', fontsize=16)
        axes[j, 1].set_ylabel(f'{parameter}', fontsize=16)
        axes[j, 1].tick_params(labelsize=14)