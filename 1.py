import torch

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set the device to CUDA
        device = torch.device("cuda")
        print("CUDA is available. Using GPU...")
        
        # Create a tensor and move it to the GPU
        hello_tensor = torch.tensor([ord(c) for c in "Hello, CUDA!"], device=device)
        
        # Transfer the tensor back to the CPU and convert to a string
        hello_string = ''.join([chr(int(c)) for c in hello_tensor.cpu()])
        
        print(hello_string)
    else:
        print("CUDA is not available. Using CPU...")
        print("Hello, CUDA!")

if __name__ == "__main__":
    main()
