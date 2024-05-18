import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import scipy
import time

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

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
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
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

# The data we are going to feed into the dataset are going to be 227x227 images
transformationTraining = transforms.Compose([
  transforms.RandomRotation(90),
  transforms.RandomResizedCrop((224,224)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomVerticalFlip(),
  transforms.ColorJitter(brightness=0.4,saturation=0.4,contrast=0.4,hue=0.15),
  transforms.ToTensor(),
  transforms.Normalize(mean = meanValues, std = stdValues)
])

transformationVal = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.CenterCrop((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean = meanValues, std = stdValues)
])

print("Download training data")
training_data = datasets.Flowers102(
  root="data",
  split="train",
  download=True,
  transform=transformationTraining
)
print("Finished downloading training data")

print("Downloading validation data")
validation_data = datasets.Flowers102(
  root="data",
  split="val",
  download=True,
  transform=transformationVal
)
print("Finished downloading validation data")

# Create data loaders.
batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
validation_dataloader = DataLoader(validation_data,batch_size=batch_size)

# Find which device we can run the project on
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif torch.backend.mps.is_available():
  device = "mps"

print(f"Using {device} device")

def train(dataloader, model, loss_func, optimizer):
  model.train()
  for batch, (X,y) in enumerate(dataloader):
    X,y = X.to(device), y.to(device)
    # Compute prediction error
    pred = model(X)
    loss = loss_func(pred,y)
    # Back propagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

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

print("Creating model")
model = NeuralNet().to(device)
print("Model created")

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.001, momentum = 0.9)

print("Starting training")
trainingStart = time.time()
epochCounter=0
bestTrainScore = 0
bestValScore = 0
bestEpoch = 0
bestTime = 0

trainScores = []
valScores = []
epochCounters = []

while epochCounter<899:
  for t in range(20):
      epochCounter+=1
      train(train_dataloader, model, loss_func, optimizer)
  hoursSinceStart = round((time.time() - trainingStart)/3600,2)
  print(f"Epoch {epochCounter} finished at {hoursSinceStart} hours")
  print("Learning rate:", optimizer.param_groups[0]['lr'])
  trainScore = test(train_dataloader,model,loss_func)
  valScore = test(validation_dataloader, model, loss_func)

  trainScores.append(trainScore)
  valScores.append(valScore)
  epochCounters.append(epochCounter)

  print(f"Accuracy on training set: {trainScore} +++ Accuracy on validation set: {valScore}")
  if(valScore>bestValScore):
     bestValScore = valScore
     bestTrainScore = trainScore
     bestEpoch = epochCounter
     bestTime = hoursSinceStart
     torch.save(model.state_dict(), f'model_weights.pth')
  print(f"The best epoch so far was epoch {bestEpoch} at {bestTime} hours.\n"+
        f"It achieved a train score of {bestTrainScore} and a val score of {bestValScore}\n")
print("Training finished")
