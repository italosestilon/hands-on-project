import torch
import torch.nn as nn

import math

#extends Module to define our very simple neural network
class Net(nn.Module):

  def __init__(self, num_classes):
    super(Net, self).__init__()

    #defining feature extractor
    self.feature_extractor = nn.Sequential(
        #defining convolutional layer
        nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1, bias=True),
        #defining activation layer
        nn.ReLU(),
        #defining pooling layer
        nn.MaxPool2d(kernel_size=(3, 3), stride=2)
    )

    #defining classifier
    self.classifier = nn.Sequential(
        #defining a linear layer that reduces from 5408 features to 4096 features
        nn.Linear(5408, 4096),
        #defining activation layer
        nn.ReLU(),
        #defining linear layer as decision layer
        nn.Linear(4096, num_classes),
    )

    #initialize weights
    self._initialize_weights()

  def forward(self, x):
      #extracts features
      x = self.feature_extractor(x)
      #transforms outputs into a 2D tensor
      x = torch.flatten(x, start_dim=1)
      #classifies patterns
      y = self.classifier(x)
  
      return y
  
  def _initialize_weights(self):
    #for each submodule of our network
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            #get the number of elements in the layer weights
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

            #initialize layer weights with random values generated from a normal
            #distribution with mean = 0 and std = sqrt(2. / n))
            m.weight.data.normal_(mean=0, std=math.sqrt(2. / n))

            if m.bias is not None:
                #initialize bias with 0 (why?)
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #initialize layer weights with random values generated from a normal
            #distribution with mean = 0 and std = 1/100
            m.weight.data.normal_(mean=0, std=0.01)

            #initialize bias with 0 (why?)
            m.bias.data.zero_()