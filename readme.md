# Anime Avatar Generator using Diffusion Algorithm

This project implements an anime face generator using a diffusion model. The model is trained to generate high-quality anime faces from random noise.


## Requirements

To install the required packages, run:

    pip install -r requirements.txt

## Training
To train the model, run:

    python diffusion/train.py

The training script uses the following components:
UNet model defined in model.py
DiffusionModel defined in diffusion.py
ImageSet dataset class defined in data.py
sample_x_t and timestep_embedding functions from process.py

## Generating Anime Faces
To generate anime faces, use the generate method of the DiffusionModel class. You can run the generation script:

    python generate.py

## File Descriptions
model.py: Contains the implementation of the UNet model and other neural network components.
diffusion.py: Contains the implementation of the DiffusionModel class.
data.py: Contains the ImageSet dataset class for loading and transforming images.
process.py: Contains utility functions for processing, including timestep embedding and noise sampling.
train.py: Training script for the diffusion model.
generate.py: Script for generating anime faces using the trained model.
