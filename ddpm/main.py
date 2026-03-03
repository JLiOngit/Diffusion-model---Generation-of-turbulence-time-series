import argparse
from dataclasses import dataclass, field
import torch as th
import matplotlib.pyplot as plt

from utils.datasets import *
from utils.plots import *
from .unet import *
from .gaussian_diffusion import *
from .training import *
from .sampling import *


@dataclass
class Config:
    # Dataset parameters
    DATASET_PATH: str = "datasets/Lagr_u1c_diffusion-demo.h5"
    NORMALIZED: bool = True
    TRAINING_SIZE: float = 0.8
    VALIDATION_SIZE: float = 0.1
    # Model hyperparameters
    INPUT_CHANNELS: int = 1
    MODEL_CHANNELS: int = 64
    OUTPUT_CHANNELS: int = 1
    NUM_RES_BLOCKS: int = 3
    NUM_HEADS: int = 2
    ATTENTION_RESOLUTION: list[int] = field(default_factory=lambda: [4, 8])
    CHANNEL_MULT: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # Diffusion parameters
    DIFFUSION_STEPS: int = 800
    SCHEDULE_NAME: str = 'linear'
    DEVICE: th.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EMA_RATE = 0.97
    LOSS_TYPE: str = "mse"
    NUM_TRAINING_STEPS = 500
    NUM_VALIDATION_STEPS = 25
    OUTPUT_DIRECTORY = "./trained_model"
    USE_CHECKPOINT: bool = True
    # Sampling parameters
    NUM_SAMPLES = 500


def parse_args():
    parser = argparse.ArgumentParser(description = 'Diffusion Model Training')
    parser.add_argument('--action', type=str, choices=['only_training', 'only_sampling', 'training_and_sampling'], default='training_and_sampling', help='train, sample or both')
    parser.add_argument('--num_diffusion', type=int, default=Config.DIFFUSION_STEPS, help='the number of diffusion steps')
    parser.add_argument('--schedule_name', type=str, default=Config.SCHEDULE_NAME, help='the beta schedule')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='the batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY, help='the weight decay')
    parser.add_argument('--ema_rate', type=float, default=Config.EMA_RATE, help='the ema rate')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'kl'], default=Config.LOSS_TYPE, help='MSE or Kullback-Leibler')
    parser.add_argument('--num_train', type=int, default=Config.NUM_TRAINING_STEPS, help='the number of training steps')
    parser.add_argument('--num_validation', type=int, default=Config.NUM_VALIDATION_STEPS, help='the number of validation steps')
    parser.add_argument('--num_samples', type=int, default=Config.NUM_SAMPLES, help='the number of samples')
    return parser.parse_args()


def main():
    # retrieved the config and the arguments
    cfg = Config()
    args = parse_args()
    # loaded normalized velocities
    normalized_velocities, max_val, min_val = load_data(cfg.DATASET_PATH, cfg.NORMALIZED)
    # split into training, validation and test sets 
    train, validation, test = training_test_split(normalized_velocities, training_size = cfg.TRAINING_SIZE, validation_size = cfg.VALIDATION_SIZE)
    # defined the diffusion model
    betas = get_named_beta_schedule(args.schedule_name , args.num_diffusion)
    model_diffusion = GaussianDiffusion(betas, cfg.DEVICE)

    if args.action in ['only_training', 'training_and_sampling']:
        # initialized the U-Net
        unet = UNet(cfg.INPUT_CHANNELS, cfg.MODEL_CHANNELS, cfg.OUTPUT_CHANNELS, cfg.NUM_RES_BLOCKS, 
                    cfg.NUM_HEADS, cfg.ATTENTION_RESOLUTION, cfg.CHANNEL_MULT)
        # defined the training arguments and trained the model
        train_loop = Train_loop(unet, model_diffusion, train, validation, args.batch_size, cfg.DEVICE, args.lr, args.weight_decay, 
                                args.ema_rate, args.loss_type, args.num_train, args.num_validation, cfg.USE_CHECKPOINT)
        train_losses, validation_losses, validation_steps = train_loop.run_loop(cfg.OUTPUT_DIRECTORY)
        # saved the loss plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(len(train_losses)), train_losses, label='Training Loss')
        ax.plot(validation_steps, validation_losses, label='Validation Loss')
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training and Validation Loss Evolution")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig("./plots/train_validation_losses.png", dpi=300)

    if args.action in ['only_sampling', 'training_and_sampling']:
        # verified the output directory exists
        if not os.path.exists(cfg.OUTPUT_DIRECTORY):
            raise FileNotFoundError(f"The output directory {cfg.OUTPUT_DIRECTORY} does not exist.")
        # verified at least one checkpoint is stored in the output directory 
        checkpoints = [d for d in os.listdir(cfg.OUTPUT_DIRECTORY) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint stored in the output directory {cfg.OUTPUT_DIRECTORY}")
        # retrieved the best checkpoint and loaded the model
        checkpoints.sort(key=lambda x: int(os.path.splitext(x)[0].split("-")[1]))
        checkpoint_path = os.path.join(cfg.OUTPUT_DIRECTORY, checkpoints[-1])
        checkpoint = th.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        trained_model = UNet(cfg.INPUT_CHANNELS, cfg.MODEL_CHANNELS, cfg.OUTPUT_CHANNELS, cfg.NUM_RES_BLOCKS, 
                    cfg.NUM_HEADS, cfg.ATTENTION_RESOLUTION, cfg.CHANNEL_MULT)
        trained_model.load_state_dict(checkpoint["model"])
        trained_model.eval()
        # generated samples using the denoising process
        generated_samples = []
        with th.no_grad(): 
            for i in range(args.num_samples):
                print(f"Generating sample {i+1}/{args.num_samples}")
                noise = th.randn_like(train[0].unsqueeze(0), device=cfg.DEVICE)
                sample_loop = Sample_loop(trained_model, betas, noise, device=cfg.DEVICE)
                V0_sample = sample_loop.run_loop_last_sample()
                generated_samples.append(V0_sample.detach().cpu().numpy())
        generated_samples = np.concatenate(generated_samples, axis=0)
        # denormalized the samples to get actual velocities
        generated_velocities = denormalize(generated_samples, max_val, min_val)
        # calculated and saved the pdf increments of sampled velocities
        tau_values = [10, 50, 100, 200]
        plot_pdf_increments(generated_velocities, tau_values)
        # saved the structure flatness of sampled velocities
        tau_values = [i * 10**j for j in range(3) for i in range(1,10)]
        p_values = [2, 4, 6, 8]
        plot_structure_flatness(generated_velocities, p_values, tau_values)


if __name__ == '__main__':
    main()