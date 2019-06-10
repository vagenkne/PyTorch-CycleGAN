from pathlib import *
from six.moves import urllib

from fastai.vision import *
from fastai.collab import *
from fastai.tabular import *
from fastai.metrics import error_rate

from PIL import Image
import numpy
import numpy as np
import os

import multiprocessing
#multiprocessing.set_start_method('spawn', True)

def _describe(self):
    print(f"shape: {self.shape}")
    print(f"dtyle: {self.dtype}")
    print(f"min_val: {torch.min(self)}")
    print(f"max_val: {torch.max(self)}")
    print(f"type: {type(self)}")

torch.Tensor.describe = _describe

# can't monkey patch numpy.array, don't hack too much, so do this instead:
def np_desc(a):
    if isinstance(a, (np.ndarray)):
        print(f"shape: {a.shape}")
        print(f"dtyle: {a.dtype}")
        
        print(f"content: {a}")    # this is ok, it is truncated nicely.
        print(f"type: {type(a)}")
    else:
        print("Not a numpy.ndarray")

path = untar_data(URLs.MNIST)
path.ls()

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train='training', ds_tfms=tfms, size=32,num_workers=0)

# take a quick look at X, and Y mini-batches
for (k, (x, y)) in enumerate(data.train_dl):
    print(f"batch: {k}")
    print("-----------")
    print("X:")
    x.describe()
    print("Y:")
    y.describe()
    print("====================================")
    if k > 0: break

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()    

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for (k, (X, Y)) in enumerate(data.train_dl, 0):
        # get the inputs
#         inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if k % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, k + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

plt.imshow((images[1].numpy()).astype(np.uint8).transpose((1, 2, 0)))

from torch.nn import functional as F

x_to_be_blur = x[0]; x_to_be_blur.shape