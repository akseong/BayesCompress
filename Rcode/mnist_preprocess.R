##################################################
## Project:   mnist preprocessing
## Date:      May 02, 2026
## Author:    Arnie Seong
##################################################
load(dplyr)
# mnist dataset downloaded from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

mnist_tr <- read.csv(here::here("data", "mnist_train.csv"))
mnist_te <- read.csv(here::here("data", "mnist_test.csv"))
unique(mnist_49tr[, 1])
mnist_49tr <- filter(mnist_tr, label %in% c(4, 9))
mnist_49te <- filter(mnist_te, label %in% c(4, 9))

# check:
mnist_49tr[1:4, 1]

matrix(mnist_49tr[1,-1], nrow = 28, byrow =TRUE)
matrix(mnist_49tr[2,-1], nrow = 28, byrow =TRUE)
matrix(mnist_49tr[3,-1], nrow = 28, byrow =TRUE)
matrix(mnist_49tr[4,-1], nrow = 28, byrow =TRUE)

nrow(mnist_49tr)
nrow(mnist_49te)

table(mnist_49tr[, 1])
table(mnist_49te[, 1])


colnames(mnist_49tr)[1] <- "four"
colnames(mnist_49te)[1] <- "four"
mnist_49tr[, 1] <- ifelse(mnist_49tr[, 1] == 4, 1, 0)
mnist_49te[, 1] <- ifelse(mnist_49te[, 1] == 4, 1, 0)
# scale
mnist_49tr[, -1] <- (mnist_49tr[, -1] / 255)
mnist_49te[, -1] <- (mnist_49te[, -1] / 255)

save(mnist_49tr, mnist_49te, file = (here::here("data", "mnist49.Rdata")))

# use nnf_binary_cross_entropy_with_logits as loss
# nnf_cross_entropy as loss for classifying all digits