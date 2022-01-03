# Implementation of a simple Variational Autoencoder (VAE) in torch for R

This is to explore what can be done with torch in R. Currently, this repo contains several basic implementations of variational autoencoders. We have

- `vae_mlp`: a basic variational autoencoder using MLP encoder and decoder.
- `vae_cnn`: same but using a more sophisticated convolutional neural network.
- `s_vae_mlp`: a supervised VAE regularized by a classifier on top the _latent_ variables as discussed in [^1], equation (2). 


[^1] **Joy, Tom, et al. "Capturing Label Characteristics in VAEs." International Conference on Learning Representations. 2020.**

