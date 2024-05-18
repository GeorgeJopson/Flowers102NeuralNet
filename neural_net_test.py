import torch
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import scipy
import time
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.convLayers = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 6
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 7
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 8
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 9
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 10
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 11
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 12
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 13
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Flatten()
        )

        self.linearLayers = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(4096, 102)
        )
    def forward(self, x):
        out = self.convLayers(x)
        out = self.linearLayers(out)
        return out

# For normalisation we use the mean and std values calculated for ImageNet
# This is because calculating the mean/std values from just the test set would
# be a small sample.
meanValues = [0.485,0.456,0.406]
stdValues = [0.229,0.224,0.225]

transformationTest = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.CenterCrop((227,227)),
  transforms.ToTensor(),
  transforms.Normalize(mean = meanValues, std = stdValues)
])

print("Downloading test data")
test_data = datasets.Flowers102(
  root="data",
  split="test",
  download=True,
  transform=transformationTest
)
print("Finished downloading test data")

batch_size = 32

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Find which device we can run the project on
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif torch.backend.mps.is_available():
  device = "mps"

print(f"Using {device} device")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return round(100*correct,3)

model = NeuralNet().to(device)

# Loss
loss_func = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('model_weights.pth'))

score=test(test_dataloader,model,loss_func)
print(f"Achieved an accuracy of {score}")
