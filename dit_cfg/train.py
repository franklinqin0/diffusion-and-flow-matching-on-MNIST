"""
Main training script for DiT with Classifier-Free Guidance on MNIST.
Replicates the lab_three notebook functionality.

Usage:
    python dit_cfg/train.py                      # Full MNIST DiT training
    python dit_cfg/train.py --sanity-check       # GMM sanity check only
    python dit_cfg/train.py --all                # Both sanity check + MNIST training
"""

import argparse
import math
import os
import sys

# Allow running directly: python dit_cfg/train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from matplotlib import pyplot as plt

from dit_cfg.distributions import GMM, MNISTSampler
from dit_cfg.probability_path import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from dit_cfg.simulators import EulerSimulator
from dit_cfg.models import (
    MLPConditionalVectorField,
    CFGVectorFieldODE,
    MNISTDiffusionTransformer,
)
from dit_cfg.trainer import CFGTrainer, MNISTCFGTrainer, visualize_output


def run_gmm_sanity_check(device: torch.device):
    """Sanity check: train MLP-based model on a Gaussian mixture."""
    print("=" * 60)
    print("Running GMM sanity check")
    print("=" * 60)

    # Initialize GMM
    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
    covs = torch.tensor([0.2, 0.2, 0.2])
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    gmm = GMM(means, covs, weights).to(device)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data=gmm,
        p_simple_shape=[2],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    vector_field = MLPConditionalVectorField(
        dim=2, hidden_dim=256, class_dim=2, num_classes=3,
    ).to(device)

    # Train
    trainer = CFGTrainer(path=path, eta=0.25, null_label=3)
    losses, steps = trainer.train(model=vector_field, num_steps=3000, lr=1e-3, batch_size=250)

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("GMM Sanity Check - Training Loss")
    plt.savefig("gmm_loss.png")
    plt.close()
    print("Saved training loss to gmm_loss.png")

    # Visualize
    guidance_strength = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Target
    ax = axes[0]
    x_data, _ = gmm.sample(250)
    x_data = x_data.detach().cpu().numpy()
    ax.scatter(x_data[:, 0], x_data[:, 1], s=5, marker="*")
    ax.set_title("Target")

    # Panel 2: Conditioned on each mode
    ax = axes[1]
    cfg_vf = CFGVectorFieldODE(vector_field, guidance_scale=guidance_strength, null_label=3)
    simulator = EulerSimulator(cfg_vf)

    batch_size = 250
    labels = torch.arange(3).repeat_interleave(batch_size).to(device)
    x_init = path.p_simple.sample(3 * batch_size)
    ts = torch.linspace(0, 1, 100).expand(3 * batch_size, -1).to(device)
    xs = simulator.simulate(x_init, ts, y=labels)
    for idx in range(3):
        xs_idx = xs[idx * batch_size: (idx + 1) * batch_size].detach().cpu().numpy()
        ax.scatter(xs_idx[:, 0], xs_idx[:, 1], s=5, label=f"Mode {idx}", marker="*")
    ax.legend()
    ax.set_title(f"CFG w/ Guidance Strength {guidance_strength:.2f}")

    # Panel 3: Unconditioned
    ax = axes[2]
    batch_size = 750
    labels = torch.ones(batch_size).long().to(device) * 3
    x_init = path.p_simple.sample(batch_size)
    ts = torch.linspace(0, 1, 100).expand(batch_size, -1).to(device)
    xs = simulator.simulate(x_init, ts, y=labels).detach().cpu().numpy()
    ax.scatter(xs[:, 0], xs[:, 1], s=5, marker="*")
    ax.set_title("Unguided Samples")

    plt.tight_layout()
    plt.savefig("gmm_sanity_check.png")
    plt.close()
    print("Saved GMM sanity check visualization to gmm_sanity_check.png")


def run_mnist_training(device: torch.device):
    """Train DiT with CFG on MNIST."""
    print("=" * 60)
    print("Training DiT on MNIST with Classifier-Free Guidance")
    print("=" * 60)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data=MNISTSampler(),
        p_simple_shape=[1, 32, 32],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    # Initialize model
    dit = MNISTDiffusionTransformer(
        patch_size=4,
        num_layers=8,
        dim=256,
        heads=8,
    ).to(device)

    # Initialize trainer
    trainer = MNISTCFGTrainer(path=path, eta=0.35, null_label=10, device=device)

    # Train
    losses, steps = trainer.train(model=dit, num_steps=20000, lr=0.4e-3, batch_size=256, ckpt_every=1000)

    # Plot loss
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MNIST DiT - Training Loss")
    plt.savefig("mnist_dit_loss.png")
    plt.close()
    print("Saved training loss to mnist_dit_loss.png")

    # Final visualization
    visualize_output(
        model=dit,
        path=path,
        device=device,
        samples_per_class=10,
        num_timesteps=100,
        guidance_scales=[1.0, 3.0, 5.0],
        save_path="mnist_dit_final.png",
    )
    print("Saved final visualization to mnist_dit_final.png")


def main():
    parser = argparse.ArgumentParser(description="DiT + CFG on MNIST")
    parser.add_argument("--sanity-check", action="store_true", help="Run GMM sanity check only")
    parser.add_argument("--all", action="store_true", help="Run both sanity check and MNIST training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.sanity_check:
        run_gmm_sanity_check(device)
    elif args.all:
        run_gmm_sanity_check(device)
        run_mnist_training(device)
    else:
        run_mnist_training(device)


if __name__ == "__main__":
    main()
