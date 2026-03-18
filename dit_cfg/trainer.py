import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

from .probability_path import GaussianConditionalProbabilityPath
from .models import CFGVectorFieldODE, ConditionalVectorField
from .simulators import EulerSimulator


MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.opt = None
        self.output_dir = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def checkpoint(self, step: int):
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def random_name(self) -> str:
        adjectives = ["autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark", "summer", "icy", "delicate", "quiet", "white", "cool", "spring", "winter", "patient"]
        foods = ["apple", "banana", "pear", "plum", "orange", "persimmon", "tangerine", "durian", "jackfruit", "jicama", "cantaloupe", "watermelon", "peach"]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    def train(
        self,
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        ckpt_every: Optional[int] = 500,
        run_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[float], List[int]]:
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join("runs", run_name)
        os.makedirs(self.output_dir, exist_ok=False)
        print("Initialized output directory at: " + self.output_dir)

        self.model = model
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.opt = self.get_optimizer(lr)
        self.model.train()

        for pg in self.opt.param_groups:
            pg["lr"] = 0.0

        losses: List[float] = []
        steps: List[int] = []

        pbar = tqdm(range(num_steps))
        for step in pbar:
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * float(step + 1) / float(warmup_steps)
            else:
                cur_lr = lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr

            self.opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            self.opt.step()

            losses.append(float(loss.detach().item()))
            steps.append(step)

            pbar.set_description(f"Step {step}, lr={cur_lr:.2e}, loss={loss.item():.4f}")

            if ckpt_every is not None and step % ckpt_every == 0:
                self.model.eval()
                self.checkpoint(step)
                self.model.train()

        self.model.eval()
        return losses, list(range(num_steps))


class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, eta: float, null_label: int, eps: float = 0.001, **kwargs):
        assert 0 < eta < 1
        super().__init__(**kwargs)
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, y = self.path.p_data.sample(batch_size)
        mask = torch.rand(batch_size).to(z) < self.eta
        y[mask] = self.null_label
        t = torch.rand(batch_size).to(z) * (1 - self.eps)
        x = self.path.sample_conditional_path(z, t)
        ut_theta = self.model(x, t, y)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        error = torch.sum(torch.square(ut_theta - ut_ref), dim=-1)
        return torch.mean(error)


def visualize_output(
    model: ConditionalVectorField,
    path: GaussianConditionalProbabilityPath,
    device: torch.device,
    samples_per_class: int = 10,
    num_timesteps: int = 100,
    guidance_scales: List[float] = [1.0, 3.0, 5.0],
    save_path: Optional[str] = None,
    use_tqdm: bool = True,
):
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

    for idx, w in enumerate(guidance_scales):
        ode = CFGVectorFieldODE(model, guidance_scale=w, null_label=10)
        simulator = EulerSimulator(ode)

        y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
        num_samples = y.shape[0]
        x0 = path.p_simple.sample(num_samples)

        ts = torch.linspace(0, 0.999, num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y, use_tqdm=use_tqdm)

        v_min, v_max = x1.min(), x1.max()
        x1 = (x1 - v_min) / (v_max - v_min)
        grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(0, 1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class MNISTCFGTrainer(CFGTrainer):
    """CFG Trainer with MNIST-specific checkpoint callback"""
    def __init__(self, *args, device: torch.device = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device or torch.device('cpu')

    def checkpoint(self, step: int):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_model.pt'))
        torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_opt.pt'))
        visualize_output(self.model, self.path, device=self.device, save_path=os.path.join(self.output_dir, f'step_{step:06d}_output.png'), use_tqdm=False)
