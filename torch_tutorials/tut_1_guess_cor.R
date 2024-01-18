##################################################
## Project:   https://torch.mlverse.org/start/guess_the_correlation/
## Date:      Jan 18, 2024
## Author:    Arnie Seong
##################################################

library(torch)
library(luz)
torch_tensor(1)
# install.packages("torchvision")
library(torchvision)
# remotes::install_github("mlverse/torchdatasets")
library(torchdatasets)


# get dataset
train_indices <- 1:10000
val_indices <- 10001:15000
test_indices <- 15001:20000

add_channel_dim <- function(img) img$unsqueeze(1)
crop_axes <- function(img) {
  transform_crop(
    img, 
    top = 0, 
    left = 21, 
    height = 131, 
    width = 130
  )
}

root <- file.path(tempdir(), "correlation")

# downloads & unpacks dataset, saved to `root`
# preprocessing using `crop_axes` & `add_channel_dim`
# uses indices to separate into train/validation/test sets
train_ds <- guess_the_correlation_dataset(
  # where to unpack
  root = root,
  # additional preprocessing 
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  # don't take all data, but just the indices we pass in
  indexes = train_indices,
  download = TRUE
)

valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = val_indices,
  download = FALSE
)

test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = test_indices,
  download = FALSE
)


length(train_ds)
length(valid_ds)
length(test_ds)


# And how does a single observation look like? Here is the first one:
train_ds[1]
# It’s a list of three items, the last of which we’re not interested in for our purposes.
# The second, a scalar tensor, is the correlation value, the thing we want the network to learn. The first, x, is the scatterplot: a tensor representing an image of dimensionality 130*130. But wait – what is that 1 in the shape output?
# This really is a three-dimensional tensor! The first dimension holds different channels – or the single channel, if the image has but one. In fact, the reason x came in this format is that we requested it, here:

# add_channel_dim <- function(img) img$unsqueeze(1)
# 
# train_ds <- guess_the_correlation_dataset(
#   # ...
#   transform = function(img) crop_axes(img) %>% add_channel_dim(),
#   # ...
# )


# add_channel_dim() was passed in as a custom transformation, to be applied to every item of the dataset. It calls one of torch’s many tensor operations, unsqueeze(), that adds a singleton dimension at a requested position.

# How about the second custom transformation?
  
  # crop_axes <- function(img) transform_crop(img, top = 0, left = 21, height = 131, width = 130)
# Here, we crop the image, cutting off the axes and labels on the left and bottom. These image regions don’t contribute any distinctive information, and having the images be smaller saves memory.



# # # # # # # # # # # # # # # # # # # # # # # # #
## Batches / Dataloader ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# Now, we’ve done so much work already, but you haven’t actually seen any of the scatterplots yet! The reason we’ve been waiting until now is that we want to show a bunch of them at a time, and for that, we need to know how to handle batches of data.
# 
# So let’s create a DataLoader object from the training set. We’ll soon use it to train the model, but right now, we’ll just plot the first batch.
# 
# A DataLoader needs to know where to get the data – namely, from the Dataset it gets passed –, as well as how many items should go in a batch. Optionally, it can return data in random order (shuffle = TRUE).

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)

length(train_dl)

batch <- dataloader_make_iter(train_dl) %>% dataloader_next()

dim(batch$x)
dim(batch$y)

# plot:
par(mfrow = c(8,8), mar = rep(0, 4))

#remove channel dimension
images <- as.array(batch$x$squeeze(2))


images %>%
  purrr::array_tree(1) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x)})


batch$y %>% as.numeric() %>% round(digits = 2)


valid_dl <- dataloader(valid_ds, batch_size = 64)
length(valid_dl)


test_dl <- dataloader(test_ds, batch_size = 64)
length(test_dl)



# # # # # # # # # # # # # # # # # # # # # # # # #
## Create the model ----
# # # # # # # # # # # # # # # # # # # # # # # # #

