import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



class ConvBlock(nn.Module):
    def __init__(self, batch_size: int, channels_in: int, channels_out: int):
        super(NeuralNetwork, self).__init__()
        self.conv1x1a = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1b = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1c = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        #Do batch_norm patří batch_size, nebo channels_out?
        self.batch_norm = nn.BatchNorm2d(batch_size)
        self.conv3x3 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward (self, input: Tensor) -> Tensor:
        output = self.conv1x1a(input)
        output = self.batch_norm(output)
        output = self.relu(output)

        forwarded_output = self.conv1x1b(input)
        forwarded_output = self.batch_norm(forwarded_output)

        output = self.conv3x3(output)
        output = self.batch_norm(output)
        output = self.relu(output)

        output = self.conv1x1c(output)
        output = self.batch_norm(output)

        output = torch.add(output, forwarded_output)
        output = self.relu(output)

        return (output)


class IdentityBlock(nn.Module):
    def __init__(self, batch_size: int, channels_in: int, channels_out: int):
        super(NeuralNetwork, self).__init__()
        self.conv1x1a = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1b = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1c = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        #Do batch_norm patří batch_size, nebo channels_out?
        self.batch_norm = nn.BatchNorm2d(batch_size)
        self.conv3x3 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward (self, input: Tensor) -> Tensor:
        output = self.conv1x1a(input)
        output = self.batch_norm(output)
        output = self.relu(output)

        output = self.conv3x3(output)
        output = self.batch_norm(output)
        output = self.relu(output)

        output = self.conv1x1c(output)
        output = self.batch_norm(output)
        
        output = torch.add(output, input)
        output = self.relu(output)

        return (output)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def forward(self, x):
        pass

model = NeuralNetwork().to(device)
print(model)