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
    def __init__(self, num_classes=10):
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

# The data we are going to feed into the dataset are going to be 256x256 images

transformationTraining = transforms.Compose([
  transforms.RandomRotation(30),
  transforms.RandomResizedCrop((227,227)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomVerticalFlip(),
  transforms.ColorJitter(brightness=0.2,saturation=0.2,contrast=0.2),
  # transforms.ToTensor(),
  # transforms.Normalize(mean = meanValues,std = stdValues)

  # transforms.Resize((256,256)),
  # transforms.CenterCrop((227,227)),
  transforms.ToTensor(),
  transforms.Normalize(mean = meanValues, std = stdValues)
])

transformationVal = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.CenterCrop((227,227)),
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
print("Finished validation test data")

batch_size = 32

# Create data loaders.
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
  size = len(dataloader.dataset)
  model.train()
  counter = 0
  for batch, (X,y) in enumerate(dataloader):
    counter+=1
    X,y = X.to(device), y.to(device)
    # Compute prediction error
    pred = model(X)
    loss = loss_func(pred,y)
    # Back propagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if batch % 5 == 0:
      loss,current = loss.item(),(batch+1)*len(X)
      #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return round(100*correct,3)

print("Creating model")
# Remove .to(device)
print("Model created")
batch_size = 64
learning_rate = 0.0005

model = NeuralNet().to(device)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.0005, momentum = 0.9)
# from torch.optim.lr_scheduler import _LRScheduler

# class CustomLR(_LRScheduler):
#     def __init__(self, optimizer, last_epoch=-1):
#         self.initial_lr = 0.001
#         self.lr_after_100_epochs = 0.0001
#         super(CustomLR, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < 100:
#             return [self.initial_lr] * len(self.base_lrs)
#         else:
#             return [self.lr_after_100_epochs] * len(self.base_lrs)
# scheduler = CustomLR(optimizer)

print("Starting training")
trainingStart = time.time()
epochCounter=0
bestTrainScore = 0
bestValScore = 0
bestEpoch = 0
bestTime = 0
while True:
  for t in range(20):
      epochCounter+=1
      train(train_dataloader, model, loss_func, optimizer)
      #scheduler.step()
  hoursSinceStart = round((time.time() - trainingStart)/3600,2)
  print(f"Epoch {epochCounter} finished at {hoursSinceStart} hours")
  print("Learning rate:", optimizer.param_groups[0]['lr'])
  trainScore = test(train_dataloader,model,loss_func)
  valScore = test(validation_dataloader, model, loss_func)
  print(f"Accuracy on training set: {trainScore} +++ Accuracy on validation set: {valScore}")
  if(valScore>bestValScore):
     bestValScore = valScore
     bestTrainScore = trainScore
     bestEpoch = epochCounter
     bestTime = hoursSinceStart
     torch.save(model.state_dict(), f'best_model_weights.pth')
  print(f"The best epoch so far was epoch {bestEpoch} at {bestTime} hours.\n"+
        f"It achieved a train score of {bestTrainScore} and a val score of {bestValScore}\n")
