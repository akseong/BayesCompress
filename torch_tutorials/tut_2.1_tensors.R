##################################################
## Project:   tut_2.1 TENSORS
## Date:      Feb 12, 2024
## Author:    Arnie Seong
##################################################

# https://torch.mlverse.org/technical/tensors/

# # # # # # # # # # # # # # # # # # # # # # # # #
## CREATING TENSORS ----
# # # # # # # # # # # # # # # # # # # # # # # # #
library(torch)

# 1d vector   
t1 <- torch_tensor(c(1,2, 3))
t1
## Q: why is this a float but the next one is of type Long?

# a 1d vector of integers 1-10
t2 <- torch_tensor(1:10)
t2

# boolean vector length 2
b2 <- torch_tensor(c(T,F))
b2

# 3x3 matrix
m3 <- torch_tensor(rbind(c(1,2,0), c(3,0,0), c(4, 5, 6)))
m3

m4 <- torch_tensor(matrix(1:16, ncol = 4, byrow = TRUE))
m4

# 3x3 matrix from N(0,1)
torch_randn(3,3)

# 3x3x3 tensor from N(0,1)
torch_randn(3,3,3)

# 2x4x3 tensor of 0s
torch_zeros(2, 4, 3)
# first index is depth (as opposed to in array())
array(0, dim = c(2, 4, 3))

# evenly spaced values
torch_arange(start = 10, end = 90, step = 5)

# identity
torch_eye(5)

# logarithmically spaced
torch_logspace(1, 3, steps = 10, base = 2)


# # # # # # # # # # # # 
##    data type    

# data type inferred from input
t <- torch_tensor(c(3, 5, 7))
t$dtype

t <- torch_tensor(1L)
t$dtype

# explicit dtype
t <- torch_tensor(2, dtype = torch_double())
t$dtype

# torch tensors live on a device. By default, this will be the CPU:
t$device

# put on gpu
t <- torch_tensor(2, device = "cuda")
t$device




# # # # # # # # # # # # # # # # # # # # # # # # #
## DATA TYPE CONVERSION ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# convert torch tensor to R obj
t <- torch_tensor(matrix(1:9, ncol = 3, byrow = TRUE))
as_array(t)

t <- torch_tensor(c(1, 2, 3))
as_array(t) %>% class()

t <- torch_ones(c(2, 2))
as_array(t) %>% class()

t <- torch_ones(c(2, 2, 2))
as_array(t) %>% class()

# can use as.integer() and as.matrix() BUT must switch device to CPU first.





# # # # # # # # # # # # # # # # # # # # # # # # #
## INDEXING/SLICING ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # 
##    R-like operations    

t <- torch_tensor(rbind(c(1,2,3), c(4,5,6)))
t

# a single value
t[1, 1]

# first row, all columns
t[1, ]

# first row, a subset of columns
t[1, 1:2]


#Note how, just as in R, singleton dimensions are dropped:
t <- torch_tensor(rbind(c(1:4), c(5:8)))

# 2x4
t$size() 

# just a single row: will be returned as a vector
t[1, 3:4]
t[1, 3:4]$size() 

# a single element
t[1, 1]$size()

# And just like in R, you can specify drop = FALSE to keep those dimensions:

t[1, 1:4, drop = FALSE]
t[1, 1:4, drop = FALSE]$size()
t[1, 1, drop = FALSE]$size()


# # # # # # # # # # # # 
##    Pythonic part

# Whereas R uses negative numbers to remove elements at specified positions, 
# in torch negative values indicate that we start counting from 
# the end of a tensor – with -1 pointing to its last element:

t <- torch_tensor(rbind(c(1,2,3), c(4,5,6)))
t
t[1, -1]
t[ , -2:-1] 


# When the slicing expression m:n is augmented by another colon 
# and a third number – m:n:o –, we will take every o'th item 
# from the range specified by m and n:
  
t <- torch_tensor(1:10)
t[2:10:2]
t[2:10:3]

t <- torch_randint(-7, 7, size = c(2, 2, 2))
t

t[.., 1]
t[2, ..]




# # # # # # # # # # # # # # # # # # # # # # # # #
## RESHAPING TENSORS ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # 
##    ZERO-COPY RESHAPING

t1 <- torch_randint(low = 3, high = 7, size = c(3, 3, 3))
t1
t1$size()

# add dimensions with unsqueeze
t2 <- t1$unsqueeze(dim = 1)
t2$size()

t3 <- t1$unsqueeze(dim = 2)
t3$size()

# remove dimensions with squeeze
t4 <- t3$squeeze()
t4$size()

# view - general.  reshape to any valid dimensionality (Same number elements)
t1 <- torch_tensor(rbind(c(1, 2), c(3, 4), c(5, 6), c(7,8)))
t1

t2 <- t1$view(c(2, 4)) # reads t1 row-wise
t2

t4 <- t1$view(c(1, 8))
t4
t4$size()

# not a new copy 
t1$storage()$data_ptr()
t2$storage()$data_ptr()
# just different metadata
# stride = how many elements traversed to arrive at next element 
# (row or column for matrix) for each dimension
t1$stride()
t2$stride()
# for array:
t1$view(c(2, 2, 2))$stride()



# # # # # # # # # # # # 
##    RESHAPE WITH COPY

t1 <- torch_tensor(rbind(c(1, 2), c(3, 4), c(5, 6)))
t1$view(6)

# but the following throws an error b/c transpose operation
# is zero-copy --- just modifies metadata.  Thus
# t2 already carries information that it should not be read
# in physical (t1) order.
t2 <- t1$t()
t2$view(6)

# so call contiguous() first to create new tensor
t3 <- t1$t()$contiguous()
t3$view(6)

# can also use reshape() --- zero-copy if possible (like view())
# but will create new copy if needed
t2$reshape(6)




# # # # # # # # # # # # # # # # # # # # # # # # #
## OPERATIONS ON TENSORS ----
# # # # # # # # # # # # # # # # # # # # # # # # #

t1 <- torch_tensor(rbind(c(1, 2), c(3, 4), c(5, 6)))
t2 <- t1$clone()

t1$add(t2)
t1 #t1 not modified

# HOWEVER, tensor method variants for MUTATING operations have trailing underscore
t1$add_(t2)
t1
# t1 modified!



# # # # # # # # # # # # # # # # # # # # # # # # #
## BROADCASTING ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# perform operations on tensors with shapes that don't match
t1 <- torch_randn(c(3,5))
t1 + 22

t1 <- torch_randn(c(3,5))
t1 + torch_tensor(c(1:5))

t1 <- torch_zeros(c(3,5))
t2 <- torch_tensor(c(1:5))
t1 + t2

# t1 is broadcast along rows, t2 broadcast along columns
t1 <- torch_tensor(1:5)$reshape(c(1, 5))
t2 <- torch_tensor(1:3)$reshape(c(3, 1))
t1$add(t2)

# computing outer product via broadcasting:
t1 <- torch_tensor(c(0, 10, 20, 30))
t2 <- torch_tensor(c(1, 2, 3))
t1$view(c(4,1)) * t2

