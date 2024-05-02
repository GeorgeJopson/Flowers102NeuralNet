import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import scipy
import time

# The average height x width of each image is 540x630 so I am going to resize all images
# to those dimensions
transformation = transforms.Compose([
  transforms.Resize((540,620)),
  transforms.ToTensor()
])

print("Download training data")
training_data = datasets.Flowers102(
  root="data",
  split="train",
  download=True,
  transform=transformation
)
print("Finished downloading training data")

print("Downloading test data")
test_data = datasets.Flowers102(
  root="data",
  split="test",
  download=True,
  transform=transformation
)
print("Finished downloading test data")

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

print("Done!")