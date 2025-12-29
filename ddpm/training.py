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
                 learning_rate=1e-4,
                 weight_decay=1e-2,
                 ema_rate=0.97,
                 loss_type='mse',
                 num_steps=1000):
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
        self.num_steps = num_steps
        self.step = 0
        self.optimizer = AdamW(list(self.model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler =th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=1e-7)

    def update_ema(self):
        with th.no_grad():
            for ema_params, model_params in zip(self.model_ema.parameters(), self.model.parameters()):
              ema_params.data = ema_params.data.to(th.float32)
              ema_params.data.mul_(self.ema_rate).add_(model_params.data.to(th.float32), alpha=1-self.ema_rate)

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

    def run_loop(self, save_model_directory, validation_step):
        print(f"Training start of {self.num_steps} steps on device {self.device}")
        train_loader = loader(self.train, self.batch_size, shuffle=True, infinite=True)
        train_losses = []
        validation_losses = []
        best_loss = float('inf')
        while self.step < self.num_steps:
            train_batch = next(train_loader)
            train_loss = self.training_step(train_batch)
            train_losses.append(train_loss)
            lr_current = self.lr_scheduler.get_last_lr()[0]
            if self.step % validation_step == 0:
                validation_loader = loader(self.validation, self.batch_size, shuffle=True, infinite=False)
                validation_loss = self.validation_step(validation_loader)
                validation_losses.append(validation_loss)
                print(f"iteration {self.step} | training loss: {train_loss:.4f} | validation loss: {validation_loss:.4f} | learning rate: {lr_current:.4e}")
                if save_model_directory is not None and validation_loss < best_loss:
                    best_loss = validation_loss
                    print(f"Save the model : validation loss: {validation_loss:.4f}")
                    model_path = os.path.join(save_model_directory, "best_unet.pth")
                    th.save(self.model.state_dict(), model_path)
            else :
                print(f"iteration {self.step} | training loss : {train_loss:.4f} | learning rate: {lr_current:.4e}")
            self.step += 1
        print("Training end")
        return train_losses, validation_losses