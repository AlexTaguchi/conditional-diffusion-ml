# Add root to python path
import sys
sys.path.append('..')

# Import modules
from conditional_diffusion_ml import Diffusion
import torch
from torchvision import datasets, transforms

# Set training parameters
batch_size = 32

# Initialize data loader
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
    ])),
    batch_size=batch_size,
    shuffle=True
)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = Diffusion([32, 32], 1, classes=10, device=device)
diffusion.train(dataloader=data_loader)
