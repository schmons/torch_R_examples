# Implementation of a simple Variational Autoencoder (VAE) in torch for R

This is to explore what can be done with torch in R. Currently, this repo contains several basic implementations of variational autoencoders. We have

- `vae_mlp`: a basic variational autoencoder using MLP encoder and decoder.
- `vae_cnn`: same but using a more sophisticated convolutional neural network.
- `s_vae_mlp`: a supervised VAE regularized by a classifier on top the _latent_ variables as discussed in [^bignote], equation (2). 


[^bignote]: **Joy, T., Schmon, S., Torr, P., Siddharth, N., & Rainforth, T. (2020, September). [Capturing Label Characteristics in VAEs.](https://openreview.net/forum?id=wQRlSUZ5V7B) In International Conference on Learning Representations.** 

