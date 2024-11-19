import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.io as io
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from volumeRendering import *
from nerfData import NerfDataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def posEncode(x, encScale):
    pows = torch.pow(2, torch.arange(0, encScale))
    products = x.unsqueeze(-1) * pows
    products = products.view(*(x.shape[:-1]), 3*encScale)

    sins = torch.sin(2*math.pi * products)
    coss = torch.cos(2*math.pi * products)
    encX = torch.stack((coss, sins), dim=-1).reshape(*(x.shape[:-1]), 6*encScale) # interleave sins and coss
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
        # we run the training loop on a batch of rays. the renderer turns this into a 2d batch, with the second dimension being the samples along each ray
        # flatten these into one dimension before passing to the model
        xShape = x.shape
        x = x.flatten(end_dim=-2) 
        x = posEncode(x, self.encScale)
        intermediateOutput = self.fcLayers(x)

        density = nn.functional.relu(self.densityLayer(intermediateOutput))
        colour = nn.functional.sigmoid(self.colourLayer(intermediateOutput))

        # unflatten
        density = density.view(*(xShape[:-1]))
        colour = colour.view(*(xShape[:-1]), 3)
        return density, colour




def train(dataloader, model, optimiser, lossFn = nn.MSELoss()):
    model.train()
    for batch, (rays, trueCols) in enumerate(dataloader):
        rays, trueCols = rays.to(device), trueCols.to(device)

        rayOrigins, rayDirs = rays.split(3, dim=-1)
        predCols = renderRays(rayOrigins, rayDirs, model)

        loss = lossFn(predCols, trueCols) 

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 200 == 0:
            size = len(dataloader.dataset)
            loss = loss.item()
            current = (batch + 1) * len(rays)
            print(f"loss: {loss:>5f} [{current:>5d}/{size:>5d}]")


def savePredImg(model, pose, path, name):
    model.eval()
    with torch.no_grad():
        img = renderScene(model, 100, 100, 3, pose)
        
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(img.permute(2, 0, 1), f"{path}{name}")
        print("saved predicted image")

def saveModel(model, path, name):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), path + name)
    except:
        print("Failed to save model")


outpath = "novelViews/nerf/output/"
dataset = NerfDataset()
dataloader = DataLoader(dataset, 100, shuffle=True)

model = NerfNet().to(device)

optimiser = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
scheduler = lr.StepLR(optimiser, step_size=10, gamma=0.1)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t}")
    train(dataloader, model, optimiser)
    scheduler.step()
    savePredImg(model, dataset.poses[1], outpath + "img/", f"{t}.png")
    saveModel(model, outpath + "model/", "model.pth")
    