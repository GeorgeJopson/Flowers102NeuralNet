import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import scipy
import time

# For normalisation we use the mean and std values calculated for ImageNet
# This is because calculating the mean/std values from just the test set would
# be a small sample. 
meanValues = [0.485,0.456,0.406]
stdValues = [0.229,0.224,0.225]

# The data we are going to feed into the dataset are going to be 256x256 images

transformationTraining = transforms.Compose([
  transforms.RandomRotation(30), 
  transforms.Resize(250),
  transforms.RandomResizedCrop(227),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(),
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
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(3, 4, kernel_size=11,stride=4,padding=(2,2)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.ReLU(),

      nn.Conv2d(4,12,kernel_size=5,stride=1,padding=(2,2)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.ReLU(),

      nn.Conv2d(12,24,kernel_size=3,stride=1,padding=(1,1)),
      nn.ReLU(),

      nn.Conv2d(24,16,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),

      nn.Conv2d(16,16,kernel_size=3,stride=1,padding=(1,1)),
      nn.ReLU(),

      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.ReLU(),

      nn.AdaptiveAvgPool2d(output_size=(6,6)),

      nn.Flatten(),
      nn.Dropout(),
      nn.Linear(576,256),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(256,256),
      nn.ReLU(),
      nn.Linear(256,102)
    )
  def forward(self,x):
    logits = self.layers(x)
    return logits

print("Creating model")  
# Remove .to(device)
model = NeuralNetwork().to(device)
print("Model created")

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.05)

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Starting training")

epochCounter=0
while True:
  for t in range(20):
      epochStartTime = time.time()
      epochCounter+=1
      #print(f"Epoch {epochCounter}\n-------------------------------")
      #print("Training")
      train(train_dataloader, model, loss_func, optimizer)
      epochEndTime = time.time()
      #print(f"Epoch length(mins): {(epochEndTime-epochStartTime)/60}")
  print(f"Epoch {epochCounter}")
  print("Testing on Training Set")
  test(train_dataloader,model,loss_func)
  print("Testing on Validation Set")
  test(validation_dataloader, model, loss_func)

  torch.save(model.state_dict(), f'model_weights{epochCounter}.pth')