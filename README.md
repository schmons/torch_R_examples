# Implementation of a simple Variational Autoencoder (VAE) in torch for R

This is to explore what can be done with [torch for R](https://torch.mlverse.org/). Currently, this repo contains several basic implementations of variational autoencoders. We have

- `vae_mlp`: a basic variational autoencoder using MLP encoder and decoder.
- `vae_cnn`: same but using a more sophisticated convolutional neural network.
- `s_vae_mlp`: a (fully) supervised VAE regularized by a classifier on top the _latent_ variables. This is not the "standard" supervised VAE but instead follows ideas of Joy et. al. (2021)[^bignote], equation (2). This isn't the best way to do (semi-)supervised variational inference. A better version would be [CCVAE](https://github.com/thwjoy/ccvae), also introduced by Joy et. al. (2021)[^bignote], Section 4.2. (I might come back to implement this when I find the time.)

## Dependencies

This package is based on [torch for R](https://torch.mlverse.org/). In addition, to load the `MNIST` dataset the code uses the [`dslab`](https://CRAN.R-project.org/package=dslabs) package. Some code also requires the `ggsci` package for color palattes.  

## Usage

The `R` files can be run in an IDE of choice such as `RStudio`.

## Latent dimensions

The variable `latent_dim` at the beginning denotes the dimension of the latent variables. If `d=2` the code will plot the latent variables created color-coded by the associated labels. This is particularly interesting for the supervised VAE.

_Note_: The focus here was to build a working prototype, so the performance of each one of them is likely far from optimal and can be improved.


[^bignote]: **Joy, T., Schmon, S., Torr, P., Siddharth, N., & Rainforth, T. (2021). [Capturing Label Characteristics in VAEs.](https://openreview.net/forum?id=wQRlSUZ5V7B) In International Conference on Learning Representations.** 

