import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import scipy
import time

normaliseCalcTransformation = transforms.Compose([
  transforms.Resize((270,310)),
  transforms.ToTensor(),
])
training_data = datasets.Flowers102(
  root="data",
  split="train",
  download=True,
  transform=normaliseCalcTransformation
)
# [R,G,B]
meanValues = [0,0,0]
stdValues = [0,0,0]
for images, _ in training_data:
   for i in range(3):
      meanValues[i]+=torch.mean(images[i,:,:])
      stdValues[i]+=torch.std(images[i,:,:])
meanValues = [m/len(training_data) for m in meanValues]
stdValues = [s/len(training_data) for s in stdValues]

# The average height x width of each image is 540x620 so I am going to resize all images
# to those dimensions
transformationTraining = transforms.Compose([
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
  transforms.Resize((270,310)),
  transforms.ToTensor(),
  transforms.Normalize(mean = meanValues, std = stdValues)
])

transformationTest = transforms.Compose([
  transforms.Resize((270,310)),
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

print("Downloading test data")
test_data = datasets.Flowers102(
  root="data",
  split="test",
  download=True,
  transform=transformationTest
)
print("Finished downloading test data")

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

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
      nn.Conv2d(3, 4, kernel_size=3,padding=1,padding_mode="replicate"),
      nn.ReLU(),
      nn.Conv2d(4,8,kernel_size=3,stride=2,padding=1,padding_mode="replicate"),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Flatten(),
      nn.Linear(41272, 128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128,64),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(64, 102)
    )
  def forward(self,x):
    logits = self.layers(x)
    return logits

print("Creating model")  
# Remove .to(device)
model = NeuralNetwork().to(device)
print("Model created")

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train(dataloader, model, loss_func, optimizer):
  size = len(dataloader.dataset)
  model.train()
  counter = 0
  for batch, (X,y) in enumerate(dataloader):
    counter+=1
    print(batch)
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
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  
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
print("Hello")

epochCounter=0
while epochCounter<=200:
  for t in range(20):
      epochStartTime = time.time()
      epochCounter+=1
      print(f"Epoch {epochCounter}\n-------------------------------")
      print("Training")
      train(train_dataloader, model, loss_func, optimizer)
      epochEndTime = time.time()
      print(f"Epoch length(mins): {(epochEndTime-epochStartTime)/60}")
  print("Testing on Training Set")
  test(train_dataloader,model,loss_func)
  print("Testing on Test Set")
  test(test_dataloader, model, loss_func)

  torch.save(model.state_dict(), f'model_weights{epochCounter}.pth')