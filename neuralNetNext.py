import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import scipy
import time

startTime = time.time()

# For normalisation we use the mean and std values calculated for ImageNet
# This is because calculating the mean/std values from just the test set would
# be a small sample. 
meanValues = [0.485,0.456,0.406]
stdValues = [0.229,0.224,0.225]

# The data we are going to feed into the dataset are going to be 256x256 images

transformationTraining = transforms.Compose([
  transforms.Resize((227,227)),
  transforms.ToTensor(),
  transforms.Normalize(mean = meanValues,std = stdValues)
])

transformationVal = transforms.Compose([
  transforms.Resize((227,227)),
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

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data,batch_size=batch_size)

# Find which device we can run the project on
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif torch.backend.mps.is_available():
  device = "mps"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
  def __init__(self,widthScale):
    # We use 4 and 256 as good base starting point to scale off of
    convLayerScale = 4 * widthScale
    linearLayerScale = 256 * widthScale
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(3, int(round(convLayerScale*1)), kernel_size=11,stride=4,padding=(2,2)),
      nn.BatchNorm2d(int(round(convLayerScale*1))),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),

      nn.Conv2d(int(round(convLayerScale*1)),int(round(convLayerScale*(3))),kernel_size=5,stride=1,padding=(2,2)),
      nn.BatchNorm2d(int(round(convLayerScale*3))),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),

      nn.Conv2d(int(round(convLayerScale*(3))),int(round(convLayerScale*6)),kernel_size=3,stride=1,padding=(1,1)),
      nn.BatchNorm2d(int(round(convLayerScale*6))),
      nn.ReLU(),

      nn.Conv2d(int(round(convLayerScale*6)),int(round(convLayerScale*4)),kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(int(round(convLayerScale*4))),
      nn.ReLU(),

      nn.Conv2d(int(round(convLayerScale*4)),int(round(convLayerScale*4)),kernel_size=3,stride=1,padding=(1,1)),
      nn.BatchNorm2d(int(round(convLayerScale*4))),
      nn.ReLU(),

      nn.MaxPool2d(kernel_size=3,stride=2),

      nn.Conv2d(int(round(convLayerScale*4)), int(round(convLayerScale*8)), kernel_size=3, stride=1, padding=(1, 1)),
      nn.BatchNorm2d(int(round(convLayerScale*8))),
      nn.ReLU(),

      nn.Conv2d(int(round(convLayerScale*8)), int(round(convLayerScale*8)), kernel_size=3, stride=1, padding=(1, 1)),
      nn.BatchNorm2d(int(round(convLayerScale*8))),
      nn.ReLU(),

      nn.AdaptiveAvgPool2d(output_size=(6,6)),

      nn.Flatten(),

      nn.Linear(6*6*int(round(convLayerScale*(8))),int(round(linearLayerScale))),
      nn.ReLU(),
      nn.Dropout(),

      nn.Linear(round(linearLayerScale),round(linearLayerScale)),
      nn.ReLU(),
      nn.Dropout(),

      nn.Linear(round(linearLayerScale),round(linearLayerScale/2)),
      nn.ReLU(),
      nn.Dropout(),

      nn.Linear(round(linearLayerScale/2),102)
    )
  def forward(self,x):
    logits = self.layers(x)
    return logits

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return round(100*correct,1)

print("Starting training")

scaleFactor = 1
print("Creating model")  
model = NeuralNetwork(scaleFactor).to(device)
print("Model created")

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

epochCounter=0
bestTrainScore = 0
bestValScore = 0
bestEpoch = 0
bestTime = 0
while True:
  for t in range(5):
      epochStartTime = time.time()
      epochCounter+=1
      train(train_dataloader, model, loss_func, optimizer)
      epochEndTime = time.time()
  epochTime = time.time()
  hoursSinceStart = round((epochTime - startTime)/3600,2)
  print(f"Epoch {epochCounter} - at {hoursSinceStart} hours")
  print("Testing on Training Set")
  trainScore = test(train_dataloader,model,loss_func)
  print("Testing on Validation Set")
  valScore = test(validation_dataloader, model, loss_func)
  if(valScore>bestValScore):
     bestValScore = valScore
     bestTrainScore = trainScore
     bestEpoch = epochCounter
     bestTime = hoursSinceStart
  print(f"The best epoch so far was epoch {bestEpoch} at {bestTime} hours.\n"+
        f"It achieved a train score of {bestTrainScore} and a val score of {bestValScore}")
  torch.save(model.state_dict(), f'best_model_weights.pth')