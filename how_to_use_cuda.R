remotes::install_github("mlverse/torch")

torch::install_torch(type = "cuda", version = "12.8")

torch::cuda_is_available()
torch::cuda_device_count()
torch::cuda_current_device()


if (cuda_is_available()) {
  torch::set_default_device("cuda")
  message("Default tensor device set to GPU (CUDA).")
} else {
  message("CUDA not available. Default tensor device remains CPU.")
}


if (torch::cuda_is_available() & use_cuda){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}



install.packages("torch")

# Create a new tensor
library(torch)
torch_set_default_tensor_type

eps <- torch_tensor(1, device = "cuda")
t1 <- torch_randn(3, 3, device = "cuda")
t2 <- torch_randn(3, 3, device = "cuda")
t3 <- t1 + t2 + eps


# Check its device
print(t3$device)

# put tensor on GPU if not set by default
x <- torch_tensor(c(1, 2, 3), device = "cuda")

# Moving a tensor
cpu_tensor <- torch_tensor(c(4, 5, 6))
gpu_tensor <- cpu_tensor$to(device = "cuda")

# Moving a model
model <- nn_module(...) # Your defined neural network model
model$to(device = "cuda")