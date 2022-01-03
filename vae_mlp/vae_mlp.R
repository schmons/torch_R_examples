# A VAE using an MLP
# 
# Author: Sebastian Schmon, 2022
# 

# This is for loading MNIST
library("dslabs")

mnist <- read_mnist()

#####################################################


library(torch)

# Set VAEs latent dimension
latent_dim <- 2

# Define encoder and decoder network
encoder <- nn_module(
  "encoder",
  initialize = function(in_features, out_features){
    
    self$modelFit <- nn_sequential(nn_linear(in_features, 400),
                                   nn_relu(),
                                   nn_linear(400, 20),
                                   nn_relu()
                                   )
    
    self$linear1 <- nn_linear(20, out_features)
    self$linear2 <- nn_linear(20, out_features)
    
  },
  forward = function(x){
    hidden = self$modelFit(x)
    output1 = self$linear1(hidden)
    output2 = self$linear2(hidden)
    
    list(output1, output2)
  }
)

decoder <- nn_module(
  "decoder",
  initialize = function(in_features, out_features){
    
    self$modelFit <- nn_sequential(nn_linear(in_features, 400),
                                   nn_relu(),
                                   nn_linear(400,out_features),
                                   nn_sigmoid())
    
  },
  forward = function(x){
    self$modelFit(x)
  }
)

# Define VAE model using encoder and decoder from above
vae_module <- nn_module(

  initialize = function(latent_dim=10) {

    self$latent_dim = latent_dim
    self$encoder <- encoder(28*28, latent_dim)
    self$decoder <- decoder(latent_dim, 28*28)
  
  },
  
  forward = function(x) {
    f <- self$encoder(x)
    mu <- f[[1]]
    log_var <- f[[2]]
    z <- mu + torch_exp(log_var)*torch_randn(c(dim(x)[1], self$latent_dim))
    reconst_x <- self$decoder(z)
    
    list(reconst_x, mu, log_var)
  }
  
)


mnist_dataset <- dataset(
  
  name = "mnist_dataset",
  
  initialize = function() {
    self$data <- self$mnist_data()
  },
  
  .getitem = function(index) {
    
    x <- self$data[index, ]

    x
  },
  
  .length = function() {
    self$data$size()[[1]]
  },
  
  mnist_data = function() {
    
    input <- torch_tensor(mnist$train$images/255) 
    input
  }
)

#Initialize the VAE module with latent dimension as specified
vae <- vae_module(latent_dim=latent_dim)

# Dataloader
dl <- dataloader(mnist_dataset(), batch_size = 250, shuffle = TRUE, drop_last=TRUE)

# Optimizer. Note that a scheduler and/or a different learning rate could improve performance
optimizer <- optim_adam(vae$parameters, lr = 0.001)

epochs = 30  # Number of full epochs (passes through the dataset)

# This is just changing graph parameters for later
par(mfrow=c(5, 4), mai=rep(0, 4))


# Optimization loop

for(epoch in 1:epochs) {
  
  l = 0
  
  coro::loop(for (b in dl) {  # loop over all minibatches for one epoch

    forward = vae(b)
    
    # likelihood part of the loss
    loss = nn_bce_loss(reduction = "sum")
    mu = forward[[2]]
    log_var = forward[[3]]
    
    # KL part of the loss
    kl_div =  1 + log_var - mu$pow(2) - log_var$exp()
    kl_div_sum = - 0.5 *kl_div$sum()
    
    # Full loss
    output <- loss(forward[[1]], b) + kl_div_sum
    
    #  
    l = l + output
    # 
    optimizer$zero_grad()
    output$backward()
    optimizer$step()

  })
  
  cat(sprintf("Loss at epoch %d: %1f\n", epoch, l))
  
  # Visualize re-constructions for 10 digits
  for(i in 1:10) {
    x_test = torch_reshape(mnist$train$images[i, ]/255, list(28, 28))
    
    mat1 = apply(matrix(as.numeric(mnist$train$images[i, ])/255, 28, 28), 1, rev)
    
    digit <- t(mat1)
    image(digit, col = grey.colors(255), axes=FALSE)
    
    recon = vae(torch_reshape(x_test, list(-1)))
    
    mat2 = torch_reshape(vae(torch_reshape(x_test, list(1, -1)))[[1]], list(28, 28))
    mat2 = matrix(as.numeric(mat2), 28, 28)
    mat2 = apply(mat2, 2, rev)
    image(t(mat2), col = grey.colors(255), axes=FALSE)
  }
  
}


# Generate new data
par(mfrow=c(2, 2))

for(i in 1:4) {
  z = torch_randn(c(1, latent_dim))
  mat = torch_reshape(vae$decoder(z), list(28, 28))
  mat = matrix(as.numeric(mat), 28, 28)
  mat = apply(mat, 2, rev)
  image(t(mat), col = grey.colors(255), axes=FALSE)
}

if(latent_dim == 2) {
  # Visualize latent 
  par(mfrow=c(1, 1))
  encoding = vae$encoder(torch_tensor(mnist$test$images/255))
  z = encoding[[1]] + torch_exp(encoding[[2]])*torch_randn(c(dim(encoding[[1]])[1], latent_dim))
  plot(z[, 1], z[, 2], pch=20, col=mnist$test$labels)
}