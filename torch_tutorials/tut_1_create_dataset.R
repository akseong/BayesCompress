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







