import torch

# Check if PyTorch is installed
print(torch.__version__)

# Check if CUDA is available
print(torch.cuda.is_available())
print(torch.version.cuda)
# Get CUDA device name
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))