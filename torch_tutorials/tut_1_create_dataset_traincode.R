##################################################
## Project:   torch tutorial: create own dataset
## Date:      Feb 11, 2024
## Author:    Arnie Seong
##################################################

# https://torch.mlverse.org/start/custom_dataset/


# `Dataset` is an R6 object that shunts data to a `Dataloader`
# `Dataset` has 3 required methods: initialize(), .getitem(i), .length()


# # # # # # # # # # # # # # # # # # # # # # # # #
## USING PALMER PENGUINS ----
# # # # # # # # # # # # # # # # # # # # # # # # #
library(torch)
library(dplyr)
library(palmerpenguins)

penguins %>% glimpse()


# initialize a torch tensor
torch_tensor(1)


# use embeddings to represent categorical data
# embedding modules expect input of type `Long`
# so initialize Long type tensor
torch_tensor(as.integer(as.numeric(as.factor("one"))))


# # # # # # # # # # # # # # # # # # # # # # # # #
## CREATE dataset OBJECT----
# # # # # # # # # # # # # # # # # # # # # # # # #

penguins_dataset <- dataset(
  
  name = "penguins_dataset",
  
  initialize = function(df) {
    
    df <- na.omit(df) 
    
    # continuous input data (x_cont)   
    x_cont <- df[ , c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year")] %>%
      as.matrix()
    self$x_cont <- torch_tensor(x_cont)
    
    # categorical input data (x_cat)
    x_cat <- df[ , c("island", "sex")]
    x_cat$island <- as.integer(x_cat$island)
    x_cat$sex <- as.integer(x_cat$sex)
    self$x_cat <- as.matrix(x_cat) %>% torch_tensor()
    
    # target data (y)
    species <- as.integer(df$species)
    self$y <- torch_tensor(species)
    
  },
  
  .getitem = function(i) {
    list(x_cont = self$x_cont[i, ], x_cat = self$x_cat[i, ], y = self$y[i])
    
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
  
)


# try out creating dataset objects
train_indices <- sample(1:nrow(penguins), 250)

train_ds <- penguins_dataset(penguins[train_indices, ])
valid_ds <- penguins_dataset(penguins[setdiff(1:nrow(penguins), train_indices), ])

length(train_ds)
train_ds$.getitem(c(1, 3, 5))



# then use datasets to instantiate dataloaders to pass data to nn
train_dl <- train_ds %>% dataloader(batch_size = 16, shuffle = TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size = 16, shuffle = FALSE)


# # # # # # # # # # # # # # # # # # # # # # # # #
## CLASSIFYING PENGUINS - MODEL ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# make separate embedding module
# has one embedding layer for every categorical feature
# inputs: cardinalities of features
# outputs: nn_embedding() for each feature
embedding_module <- nn_module(
  
  initialize = function(cardinalities) {
    
    self$embeddings = nn_module_list(
      lapply(
        cardinalities, 
        function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2))
      )
    )
    
  },
  
  forward = function(x) {
    
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    
    torch_cat(embedded, dim = 2)
  }
)



# network
net <- nn_module(
  "penguin_net",
  
  # specify layers
  initialize = function(cardinalities,
                        n_cont,
                        fc_dim,
                        output_dim) {
    
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(sum(purrr::map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + n_cont, fc_dim)
    self$output <- nn_linear(fc_dim, output_dim)
    
  },
  
  
  forward = function(x_cont, x_cat) {
    
    embedded <- self$embedder(x_cat)
    
    # concatenate embeddings and continuous vars
    all <- torch_cat(list(embedded, x_cont$to(dtype = torch_float())), dim = 2)
    
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$output() %>%
      nnf_log_softmax(dim = 2)
    
  }
)

# model
model <- net(
  cardinalities = c(length(levels(penguins$island)), length(levels(penguins$sex))),
  n_cont = 5,
  fc_dim = 32,
  output_dim = 3
)



# # # # # # # # # # # # # # # # # # # # # # # # #
## TRAIN MODEL ----
# # # # # # # # # # # # # # # # # # # # # # # # #


optimizer <- optim_adam(model$parameters, lr = 0.01)

for (epoch in 1:20) {
  
  model$train()
  train_losses <- c()  
  
  coro::loop(for (b in train_dl) {
    
    optimizer$zero_grad()
    output <- model(b$x_cont, b$x_cat)
    loss <- nnf_nll_loss(output, b$y)
    
    loss$backward()
    optimizer$step()
    
    train_losses <- c(train_losses, loss$item())
    
  })
  
  model$eval()
  valid_losses <- c()
  
  coro::loop(for (b in valid_dl) {
    
    output <- model(b$x_cont, b$x_cat)
    loss <- nnf_nll_loss(output, b$y)
    valid_losses <- c(valid_losses, loss$item())
    
  })
  
  cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f\n", epoch, mean(train_losses), mean(valid_losses)))
}


