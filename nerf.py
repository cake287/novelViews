import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.io as io
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import math
import os

import volumeRendering



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def posEncode(x, encScale):
    pows = torch.pow(2, torch.arange(0, encScale))
    products = x.unsqueeze(-1) * pows
    products = products.view(x.shape[0], 3*encScale)

    sins = torch.sin(2*math.pi * products)
    coss = torch.cos(2*math.pi * products)
    encX = torch.stack((coss, sins), dim=-1).reshape(x.shape[0], 6*encScale) # interleave sins and coss
    return encX

class NerfNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 5
        self.encScale = 6


        layers = []
        
        layers.append(nn.Linear(6*self.encScale, 256))
        for _ in range(self.depth):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(256, 256))

        self.fcLayers = nn.Sequential(*layers)
    
        self.densityLayer = nn.Linear(256, 1)
        self.colourLayer = nn.Linear(256, 3)

    def forward(self, x):
        encX = posEncode(x, self.encScale)
        intermediateOutput = self.fcLayers(encX)

        density = nn.functional.relu(self.densityLayer(intermediateOutput))
        colour = nn.functional.sigmoid(self.colourLayer(intermediateOutput))
        return density, colour





def train(dataloader, model, optimiser, lossFn = nn.MSELoss()):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = lossFn(pred, y) 

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 10000 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>5f} [{current:>5d}/{size:>5d}]")



data = np.load(R"data\nerf\tiny_nerf_data.npz")
poses = torch.tensor(data["poses"])
images = torch.tensor(data["images"])
focal = data["focal"]


model = NerfNet().to(device)

inputs = torch.tensor([
    [0.5, 1.2, 0.7],
    [-0.3, 0.4, 1.0]
])


outputs = model(inputs)
print(outputs)