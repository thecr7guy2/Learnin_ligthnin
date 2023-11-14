# CIFAR-10 Classifier with PyTorch Lightning and Wandb Integration

## Description
This project implements a CIFAR-10 image classifier using a custom ResNet-18 model in PyTorch Lightning. It features integration with Weights & Biases (wandb) for experiment tracking and hyperparameter tuning.

## Goal of the Project
The primary objective of this project is to learn PyTorch Lightning from scratch and gain a thorough understanding of experiment tracking using Weights & Biases (wandb). The project aims to demonstrate:
- How to effectively utilize PyTorch Lightning to build and train machine learning models.
- The capabilities of wandb in experiment tracking, model performance visualization, and hyperparameter tuning.
- Best practices in managing and monitoring machine learning experiments.
This serves as a practical application to understand the nuances of modern ML frameworks and experiment tracking tools.


## Installation
To install the necessary dependencies, run the following command:
```bash
pip install pytorch-lightning torchvision torchmetrics wandb
```
## Usage
The main script for training the model is train.py. It sets up the CIFAR-10 data module, model, and training logic.

## Training the model
To run the training script, execute:

```bash
python train.py
```
This will start a training session with the default hyperparameters and log metrics to wandb.

## Using Wandb Sweeps

To perform hyperparameter tuning with wandb sweeps, first create a sweep configuration file (sweep.yaml) and then initialize the sweep:

```bash
wandb sweep sweep.yaml
wandb agent [SWEEP_ID]
```

## Contributing
Contributions to this project are welcome. Please fork the repository and open a pull request with your changes.



