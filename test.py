import torch
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torch.nn as nn

from torchvision import transforms

import numpy as np

from torchvision import datasets

from Network import Net

def load_dataset(transform):
    trainset = datasets.MNIST('dataset', train=True, download=True, transform=transform)
    testset = datasets.MNIST('dataset', train=True, download=True, transform=transform)

    return trainset, testset

def get_device():
    gpu = torch.cuda.is_available()
    device = torch.device(0) if gpu else torch.device('cpu')

    return device

def load_model(filename, model):
    model.load_state_dict(torch.load(filename), strict=True)


#defining testing loop
def test(model, testload, criterion, device):
  #put model in evaluation mode
  model.eval()

  running_loss = 0.0
  running_corrects = 0.0
  n = 0

  number_of_batches = len(testload)
  
  #get a batch
  for batch_index, data in enumerate(testload, 0):
    inputs, labels = data

    #mode data to the correct device
    inputs, labels = inputs.to(device), labels.to(device)
    
    #deactivate gradients computation
    with torch.no_grad():

      #do forward through model
      outputs = model(inputs)

      #calculate loss
      loss = criterion(outputs, labels)
      
      #accumulate information to calculate accuracy
      preds = torch.max(outputs, 1)[1]
      running_loss += loss.item()*inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)
      n += outputs.size(0)
  
  test_loss = running_loss/n
  test_acc = running_corrects.double()/n
    
  print()
  print("Metrics")
  print('Loss: {:.6f} Acc: {:.6f}'.format(test_loss, test_acc))


def main():
    device = get_device()

    # prepare the data
    transform = transforms.Compose([transforms.ToTensor()])

    _, full_testset = load_dataset(transform)
    
    testload = DataLoader(full_testset, batch_size=64, shuffle=False)

    # load model
    model = Net(num_classes=10)
    load_model("model.pt", model)

    # criterion function
    criterion = nn.CrossEntropyLoss()

    # test
    test(model, testload, criterion, device)

if __name__ == "__main__":
    main()
else:
    main()
