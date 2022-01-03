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
  
  initialize = function(latent_dim) {
    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$conv1 <- nn_conv2d(1, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, latent_dim)
    self$fc3 <- nn_linear(128, latent_dim)
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      self$dropout1() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout2() %>%
      list(self$fc2(.), self$fc3(.))
    
      # output1 = self$fc2(hidden)
      # output2 = self$fc3(hidden)
      # list(output1, output2)
  }
)


# class Decoder(nn.Module):
#   def __init__(self):
#   super(Decoder, self).__init__()
# c = capacity
# self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
# self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
# self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
# 
# def forward(self, x):
#   x = self.fc(x)
# x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
# x = F.relu(self.conv2(x))
# x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
# return x

decoder <- nn_module(
  "decoder",
  initialize = function(latent_dim){
    self$fc1 <- nn_linear(latent_dim, 128)
    self$fc2 <- nn_linear(128, 128)
    self$conv1 <- nn_conv_transpose2d(128, 256, 1)
    self$conv2 <- nn_conv_transpose2d(256, 784, 1) 
  },
  forward = function(x) {
    x = self$fc1(x)
    x1 = nnf_relu(x)
    x2 = self$fc2(x1)
    x3 = nnf_relu(x2)
    x4 = torch_reshape(x3, list(x3$size(1), 64*2, 1, 1))
    x5 = self$conv1(x4)
    x6 = nnf_relu(x5)
    x7 = self$conv2(x6)
    x8 = torch_reshape(x7, list(x$size(1), -1))
    nnf_sigmoid(x8)
  }
)

# net = encoder(10)
# 
# forward = net(torch_reshape(torch_tensor(mnist$train$images[1:2, ]/255), list(2, 1, 28, 28)))
# forward[[2]]$size(1)
# 
# net2 = decoder()
# net2(forward[[2]])
# Define VAE model using encoder and decoder from above
vae_module <- nn_module(

  initialize = function(latent_dim=10) {

    self$latent_dim = latent_dim
    self$encoder <- encoder(latent_dim)
    self$decoder <- decoder(latent_dim)
  
  },
  
  forward = function(x) {
    f <- self$encoder(x)
    mu <- f[[2]]
    log_var <- f[[3]]
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

    #print(dim(b))
    #print(vae$encoder(b))
    forward = vae(torch_reshape(b, list(b$size(1), 1, 28, 28)))
    
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
    
    recon = vae(torch_reshape(x_test, list(1, 1, 28, 28)))
    
    mat2 = torch_reshape(vae(torch_reshape(x_test, list(1, 1, 28, 28)))[[1]], list(28, 28))
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

