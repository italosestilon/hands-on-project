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

#defining training loop
def train(model, trainload, criterion, optimizer, lr_scheduler, epochs, device):
  #put model in train model
  model.train()
  
  for epoch in range(epochs):
    print('-' * 20)
    print('Epoch {}/{}'.format(epoch, epochs - 1))

    running_loss = 0.0
    running_corrects = 0.0
    n = 0

    number_of_batches = len(trainload)
    
    #get a batch
    for batch_index, data in enumerate(trainload, 0):
        inputs, labels = data
        #move data to the correct device
        inputs, labels = inputs.to(device), labels.to(device)
        
        #zero gradients
        optimizer.zero_grad()

        #do forward through the neural network
        outputs = model(inputs)

        #compute the loss
        loss = criterion(outputs, labels)
        
        #compute gradients
        loss.backward()

        #update weights
        optimizer.step()

        #accumulate information to calculate accuracy
        preds = torch.max(outputs, 1)[1]
        running_loss += loss.item()*inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        n += outputs.size(0)

        print("\rBatch {}/{}".format(batch_index, number_of_batches))
    
    epoch_loss = running_loss/n
    epoch_acc = running_corrects.double()/n

    #count epochs
    lr_scheduler.step()
    
    print()
    print("Metrics")
    print('Loss: {:.6f} Acc: {:.6f}'.format(epoch_loss, epoch_acc))

def get_device():
    gpu = torch.cuda.is_available()
    device = torch.device(0) if gpu else torch.device('cpu')

    return device

def save_model(filename, model):
    torch.save(model.state_dict(), filename)


def test(model, testload, criterion, device):
  model.eval()

  running_loss = 0.0
  running_corrects = 0.0
  n = 0

  number_of_batches = len(testload)
  
  for batch_index, data in enumerate(testload, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
      outputs = model(inputs)

      loss = criterion(outputs, labels)
      
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

    full_trainset, full_testset = load_dataset(transform)

    trainset, valset, _ = random_split(full_trainset, [1000, 500, 58500])
    
    trainload = DataLoader(trainset, batch_size=64, shuffle=True)
    valload = DataLoader(valset, batch_size=64, shuffle=True)

    # prepare the optimizer
    model = Net(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(),
                      lr=0.01, momentum=0.9, weight_decay=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                      step_size=50, gamma=0.1)

    #criterion function
    criterion = nn.CrossEntropyLoss()

    # train
    train(model, trainload, criterion, optimizer, lr_scheduler, 1, device)

    # validate
    test(model, valload, criterion, device)
    
    #save model weigths for testing
    save_model("model.pt", model)

if __name__ == "__main__":
    main()
else:
    main()
