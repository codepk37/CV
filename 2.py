import torch

# %%
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Found {torch.cuda.device_count()} CUDA device(s)")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: ")
        print(f"\t{torch.cuda.get_device_properties(i)}")
else:
    print("CUDA is not available")
