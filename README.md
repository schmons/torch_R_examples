# Implementation of a simple Variational Autoencoder (VAE) in torch for R

This is to explore what can be done with torch in R. Currently, this repo contains several basic implementations of variational autoencoders. We have

- `vae_mlp`: a basic variational autoencoder using MLP encoder and decoder.
- `vae_cnn`: same but using a more sophisticated convolutional neural network.
- `s_vae_mlp`: a supervised VAE regularized by a classifier on top the _latent_ variables as discussed by Joy et. al. (2021)[^bignote], equation (2). 

_Note_: The focus here was to build a working prototype, so the performance of each one of them is likely far from optimal.


[^bignote]: **Joy, T., Schmon, S., Torr, P., Siddharth, N., & Rainforth, T. (2021). [Capturing Label Characteristics in VAEs.](https://openreview.net/forum?id=wQRlSUZ5V7B) In International Conference on Learning Representations.** 

