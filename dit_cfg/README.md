# DiT for Classifier-Free Guidance

This folder modularizes 6.s184's lab 3 Jupyter notebook.

| File | Contents |
|------|----------|
| `__init__.py` | Re-exports all public classes |
| `distributions.py` | Sampleable, LabeledSampleable, IsotropicGaussian, GMM, MNISTSampler |
| `probability_path.py` | Alpha, Beta, LinearAlpha, LinearBeta, ConditionalProbabilityPath, GaussianConditionalProbabilityPath |
| `simulators.py` | ODE, SDE, Simulator, EulerSimulator, EulerMaruyamaSimulator |
| `models.py` | ConditionalVectorField, CFGVectorFieldODE, MLP, MLPConditionalVectorField, FourierEncoder, Patchifier, Depatchifier, MHA, DiffusionTransformerLayer, DiffusionTransformer, MNISTDiffusionTransformer |
| `trainer.py` | Trainer, CFGTrainer, MNISTCFGTrainer, visualize_output |
| `train.py` | CLI entry point with --sanity-check (GMM) and --all flags |


Run with:

- python dit_cfg/train.py — full MNIST DiT training
- python train.py --sanity-check — GMM sanity check only
- python train.py --all — both
