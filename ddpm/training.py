import numpy as np
import torch as th
import os
from torch.optim import AdamW
import copy
from utils.datasets import loader


class Train_loop:

    def __init__(self,
                 model,
                 diffusion,
                 train,
                 validation,
                 batch_size,
                 device,
                 learning_rate,
                 weight_decay,
                 ema_rate,
                 loss_type,
                 num_training_steps,
                 num_validation_steps,
                 use_checkpoint):
        self.model = model.to(device)
        self.model_ema = copy.deepcopy(model).to(device)
        self.model_ema.eval()
        self.model_ema.load_state_dict(model.state_dict())
        self.diffusion = diffusion
        self.train = train
        self.validation = validation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.weight_decay = weight_decay
        self.ema_rate = ema_rate
        self.loss_type = loss_type
        self.num_training_steps = num_training_steps
        self.num_validation_steps = num_validation_steps
        self.use_checkpoint = use_checkpoint
        self.step = 0
        self.optimizer = AdamW(list(self.model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler =th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_training_steps, eta_min=1e-7)

    def update_ema(self):
        with th.no_grad():
            for ema_params, model_params in zip(self.model_ema.parameters(), self.model.parameters()):
              ema_params.data = ema_params.data.to(th.float32)
              ema_params.data.mul_(self.ema_rate).add_(model_params.data.to(th.float32), alpha=1-self.ema_rate)

    def save_checkpoint(self, output_directory, filename, train_losses=None, validation_losses=None, validation_steps=None):
        os.makedirs(output_directory, exist_ok=True)
        checkpoint = {"model": self.model.state_dict(),
                      "model_ema": self.model_ema.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "scheduler": self.lr_scheduler.state_dict(),
                      "step": self.step,
                      "train_losses": train_losses if train_losses is not None else [],
                      "validation_losses": validation_losses if validation_losses is not None else [],
                      "validation_steps": validation_steps if validation_steps is not None else []}
        checkpoint_path = os.path.join(output_directory, filename)
        th.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved : {checkpoint_path}")

    def load_checkpoint(self, output_directory):
        if not os.path.exists(output_directory):
            print(f"The output directory {output_directory} does not exist and will be created.")
            os.makedirs(output_directory, exist_ok=True)
            return None, None
        checkpoints = [d for d in os.listdir(output_directory) if d.startswith("checkpoint-")]
        if not checkpoints:
            print(f"No checkpoint stored in the output directory {output_directory}")
            return None, None
        checkpoints.sort(key=lambda x: int(os.path.splitext(x)[0].split("-")[1]))
        checkpoint_path = os.path.join(output_directory, checkpoints[-1])
        checkpoint = th.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        print(f"Checkpoint of step {self.step} has been loaded : {checkpoint_path}")
        return checkpoint_path, checkpoint

    def training_step(self, batch):
        batch = batch.to(self.device, dtype=th.float32)
        self.optimizer.zero_grad()
        t = th.randint(0, self.diffusion.num_diffusion_steps, (batch.shape[0],), device=self.device)
        noise = th.randn_like(batch)
        losses = self.diffusion.training_losses(self.model, batch, t, noise, self.loss_type)
        if self.loss_type == 'mse':
            loss = losses["mse"].mean()
        elif self.loss_type == 'kl':
            loss = losses["kl"].mean()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.update_ema()
        return loss.item()

    def validation_step(self, validation_loader):
        validation_batch_losses = []
        self.model.eval()
        for validation_batch in validation_loader:
            validation_batch = validation_batch.to(self.device, dtype=th.float32)
            with th.no_grad():
                t = th.randint(1, self.diffusion.num_diffusion_steps, (validation_batch.shape[0],), device=self.device)
                noise = th.randn_like(validation_batch).to(th.float32)
                losses = self.diffusion.training_losses(self.model_ema, validation_batch, t, noise, self.loss_type)
                if self.loss_type == 'mse':
                    validation_batch_loss = losses["mse"].mean().item()
                elif self.loss_type == 'kl':
                    validation_batch_loss = losses["kl"].mean().item()
                validation_batch_losses.append(validation_batch_loss)
        return np.mean(validation_batch_losses)

    def run_loop(self, output_directory):
        train_loader = loader(self.train, self.batch_size, shuffle=True, infinite=True)
        last_checkpoint_path = None
        train_losses = []
        validation_losses = []
        validation_steps = []
        best_loss = float('inf')
        if self.use_checkpoint:
            print(f"Checking if there is any checkpoint")
            last_checkpoint_path, last_checkpoint = self.load_checkpoint(output_directory)
            if last_checkpoint is not None:
                train_losses = last_checkpoint.get("train_losses", [])
                validation_losses = last_checkpoint.get("validation_losses", [])
                validation_steps = last_checkpoint.get("validation_steps", [])
                best_loss = min(validation_losses)
        print(f"Training start of {self.num_training_steps} steps on device {self.device}")
        while self.step < self.num_training_steps:
            self.step += 1
            train_batch = next(train_loader)
            train_loss = self.training_step(train_batch)
            train_losses.append(train_loss)
            lr_current = self.lr_scheduler.get_last_lr()[0]
            if self.step > 0 and self.step % self.num_validation_steps == 0:
                validation_loader = loader(self.validation, self.batch_size, shuffle=True, infinite=False)
                validation_loss = self.validation_step(validation_loader)
                validation_losses.append(validation_loss)
                validation_steps.append(self.step)
                print(f"iteration {self.step} | training loss: {train_loss:.4f} | validation loss: {validation_loss:.4f} | learning rate: {lr_current:.4e}")
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    if last_checkpoint_path is not None and os.path.exists(last_checkpoint_path):
                        os.remove(last_checkpoint_path)
                    self.save_checkpoint(output_directory,
                                         f"checkpoint-{self.step}.pt",
                                         train_losses,
                                         validation_losses, 
                                         validation_steps)
                    last_checkpoint_path = os.path.join(output_directory, f"checkpoint-{self.step}.pt")
            else :
                print(f"iteration {self.step} | training loss : {train_loss:.4f} | learning rate: {lr_current:.4e}")
        print("Training end")
        return train_losses, validation_losses, validation_steps