from collections import OrderedDict

import torch
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torch.nn as nn

from torchvision import transforms

import numpy as np

from torchvision import datasets

import umap

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

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

#function to get the output of each layer
def get_output_by_layer(model, x):
  #empty dict
  output_by_layer = OrderedDict()
  
  #get the input
  output_by_layer['input'] = x.clone().detach().cpu().data.numpy()

  #for each layer of the feature extractor
  for layer_name, layer in model.feature_extractor.named_children():
    #do forward through the layer
    x = layer.forward(x)
    #save the output
    output_by_layer[layer_name] = x.clone().detach().cpu().numpy()
  
  #transform features to a 2D tensor
  x = x.flatten(start_dim=1)
  for layer_name, layer in model.classifier.named_children():
    #do forward through the layer   
    x = layer.forward(x)
    #save the output
    output_by_layer["classifier-"+layer_name] = x.clone().detach().cpu().numpy()
  
  #return output by layer
  return output_by_layer

#get the outputs, and labels
def get_ouputs(model, dataload, device):
  outputs_by_layer = None
  all_labels = None

  #get a batch from the dataload
  for inputs, labels in dataload:
    #move inputs to the correct device
    inputs = inputs.to(device)
    labels = labels.clone().detach().cpu().numpy()

    #get outputs by layer
    outputs = get_output_by_layer(model, inputs)

    #save the outputs
    if outputs_by_layer is None:
      outputs_by_layer = outputs
      all_labels = labels

    else:
      for layer in outputs:
          outputs_by_layer[layer] = np.concatenate((outputs_by_layer[layer], outputs[layer]), axis=0)
      all_labels = np.concatenate((all_labels, labels))   

  return outputs_by_layer, all_labels

def projection(output_by_layer, reducer):
  projection_by_layer = OrderedDict()

  for layer in output_by_layer:
    output = output_by_layer[layer]
    output = output.reshape(output.shape[0], -1)
    embedded = reducer.fit_transform(output)

    projection_by_layer[layer] = embedded
  
  return projection_by_layer

def create_visualization(projection_by_layer, all_labels):
  
  for layer in projection_by_layer:
    embedded = projection_by_layer[layer]
  
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(embedded[:, 0], embedded[:, 1], c=all_labels, cmap=plt.get_cmap('tab10'))
    plt.axis("off")
    plt.title(layer)
    #plt.set_title(layer)
    plt.colorbar()
    plt.savefig("{}.png".format(layer))
    plt.close(fig)

def main():
    device = get_device()

    # prepare the data
    transform = transforms.Compose([transforms.ToTensor()])

    full_trainset, _ = load_dataset(transform)

    visset,_ = random_split(full_trainset, [300, 59700])
    
    visload = DataLoader(visset, batch_size=64, shuffle=False)

    # load model
    model = Net(num_classes=10)
    load_model("model.pt", model)

    # visualize

    # define dimensionality reducer
    reducer = umap.UMAP()
    #reducer = TSNE(perplexity=50)

    output_by_layer, all_labels = get_ouputs(model, visload, device)
    projection_by_layer = projection(output_by_layer, reducer)

    create_visualization(projection_by_layer, all_labels)

if __name__ == "__main__":
    main()
else:
    main()
