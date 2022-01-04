# A SSVAE using an MLP
# 
# Author: Sebastian Schmon, 2022
# 

# This is for loading MNIST
library("dslabs")

mnist <- read_mnist()

#####################################################

# Set dimension of latent variable
latent_dim = 50


library(torch)

# Define encoder and decoder network
encoder <- nn_module(
  "encoder",
  initialize = function(in_features, out_features){
    
    self$modelFit <- nn_sequential(nn_linear(in_features, 400),
                                   nn_relu(),
                                   nn_linear(400, 50),
                                   nn_relu()
                                   )
    
    self$linear1 <- nn_linear(50, out_features)
    self$linear2 <- nn_linear(50, out_features)
    
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
    
    self$modelFit <- nn_sequential(nn_linear(in_features, 200),
                                   nn_relu(),
                                   nn_linear(200,400),
                                   nn_relu(),
                                   nn_linear(400,out_features),
                                   nn_sigmoid())
    
  },
  forward = function(x){
    self$modelFit(x)
  }
)

classifier <- nn_module(
  "classifier",
  initialize = function(in_features) {
    
    self$modelFit <- nn_sequential(nn_linear(in_features, 10)
                                   )
  },
  forward = function(x){
    self$modelFit(x)
  }
)

# Define VAE model using encoder and decoder from above
ss_vae_module <- nn_module(

  initialize = function(latent_dim=50) {

    self$latent_dim = latent_dim
    self$encoder <- encoder(28*28, latent_dim)
    self$decoder <- decoder(latent_dim, 28*28)
    self$classifier <- classifier(latent_dim)
  
  },
  
  forward = function(x) {
    f <- self$encoder(x)
    mu <- f[[1]]
    log_var <- f[[2]]
    z <- mu + torch_exp(log_var$mul(0.5))*torch_randn(c(dim(x)[1], self$latent_dim))
    reconst_x <- self$decoder(z)
    prediction <- self$classifier(z)
    
    list(reconst_x, mu, log_var, prediction)
  }
  
)


mnist_dataset <- dataset(
  
  name = "mnist_dataset",
  
  initialize = function() {
    self$data <- self$mnist_data()
  },
  
  .getitem = function(index) {
    
    x <- self$data$x[index, ]
    y <- self$data$y[index]
    list(x=x, y=y)
  },
  
  .length = function() {
    self$data$x$size()[[1]]
  },
  
  mnist_data = function() {
    
    x <- torch_tensor(mnist$train$images/255) 
    y <- torch_tensor(mnist$train$labels + 1, dtype = torch_long()) # loss doesn't work if we start indexing at 0
    list(x=x, y=y)
  }
)

#Initialize the VAE module with latent dimension as specified
ss_vae <- ss_vae_module(latent_dim=latent_dim)

# Dataloader
dl <- dataloader(mnist_dataset(), batch_size = 250, shuffle = TRUE, drop_last=TRUE)

# Optimizer. Note that a scheduler and/or a different learning rate could improve performance
optimizer <- optim_adam(ss_vae$parameters, lr = 0.001)
lmbda <- function(epoch) 0.8
scheduler <- lr_multiplicative(optimizer, lr_lambda=lmbda, last_epoch = -1, verbose = FALSE)


epochs = 40  # Number of full epochs (passes through the dataset)

# This is just changing graph parameters for later


# Optimization loop

for(epoch in 1:epochs) {
  
  l = 0
  acc = 0
  
  coro::loop(for (b in dl) {  # loop over all minibatches for one epoch

    forward = ss_vae(b$x)
    
    # likelihood part of the loss
    loss = nn_bce_loss(reduction = "sum")
    classifier_loss = nn_cross_entropy_loss(reduction="sum")
    mu = forward[[2]]
    log_var = forward[[3]]
    
    # KL part of the loss
    kl_div =  1 + log_var - mu$pow(2) - log_var$exp()
    kl_div_sum = - 0.5 *kl_div$sum()

    regularizer = 100*classifier_loss(forward[[4]], b$y)

    # Full loss
    output <- loss(forward[[1]], b$x) + kl_div_sum + regularizer
    
    #  
    l = l + output
    # 
    optimizer$zero_grad()
    output$backward()
    optimizer$step()
    
    
    # Compute accuracy
    pred = torch_argmax(forward[[4]], 2)
    #print(pred == b$y)
    batch_acc = sum(as.numeric(pred == b$y))
    acc = acc + batch_acc

  })
  
  #scheduler$step()
  cat(sprintf("Loss at epoch %d: %3f | Training accuracy: %3f\n", epoch, 128*l/60000, acc/60000))
  
  # Visualize re-constructions for 10 digits
  par(mfrow=c(5, 4), mai=rep(0, 4))
  
  for(i in 1:10) {
    x_test = torch_reshape(mnist$train$images[i, ]/255, list(28, 28))

    mat1 = apply(matrix(as.numeric(mnist$train$images[i, ])/255, 28, 28), 1, rev)

    digit <- t(mat1)
    image(digit, col = grey.colors(255), axes=FALSE)

    recon = ss_vae(torch_reshape(x_test, list(1, -1)))

    mat2 = torch_reshape(ss_vae(torch_reshape(x_test, list(1, -1)))[[1]], list(28, 28))
    mat2 = matrix(as.numeric(mat2), 28, 28)
    mat2 = apply(mat2, 2, rev)
    image(t(mat2), col = grey.colors(255), axes=FALSE)

    label = torch_argmax(recon[[4]])-1 
    text(0.1,0.1, as.numeric(label), col="gray90", cex=2)
    
  }
  
  if(latent_dim == 2) {
    
    # Visualize latent (we just show the means)
    require("ggsci")
    par(mfrow=c(1, 1))
    encoding = ss_vae$encoder(torch_tensor(mnist$test$images/255))
    z = encoding[[1]] + torch_exp(encoding[[2]]$mul(0.5))*torch_randn(c(dim(encoding[[1]])[1], latent_dim))
    plot(z[, 1], z[, 2], pch=20, col=pal_d3("category10")(10)[c(mnist$test$labels+1)])
  
  }
  
}


# Generate new data
par(mfrow=c(4, 4))

for(i in 1:16) {
  z = torch_randn(c(1, latent_dim))
  mat = torch_reshape(ss_vae$decoder(z), list(28, 28))
  mat = matrix(as.numeric(mat), 28, 28)
  mat = apply(mat, 2, rev)
  image(t(mat), col = grey.colors(255), axes=FALSE)
  m = nn_log_softmax(1)
  pred = ss_vae$classifier(z)
  label = torch_argmax(pred)-1
  text(0.1,0.1, as.numeric(label), col="gray90", cex=2)
}

# Overall test accuracy
encoding = ss_vae$encoder(torch_tensor(mnist$test$images/255))
z = encoding[[1]] + torch_exp(encoding[[2]]$mul(0.5))*torch_randn(c(dim(encoding[[1]])[1], latent_dim))

pred = ss_vae$classifier(z)
pred = torch_argmax(pred, 2)

acc = mean(as.numeric(pred == mnist$test$labels+1))
acc
