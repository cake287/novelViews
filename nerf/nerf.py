import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.io as io
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr
import numpy as np
import math
import os
from time import time

from volumeRendering import *
from nerfData import NerfDataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def posEncode(x, encScale):
    pows = torch.pow(2, torch.arange(0, encScale)).to(device)
    products = x.unsqueeze(-1) * pows
    products = products.view(*(x.shape[:-1]), x.shape[-1]*encScale)

    sins = torch.sin(2*math.pi * products)
    coss = torch.cos(2*math.pi * products)
    encX = torch.stack((coss, sins), dim=-1).reshape(*(x.shape[:-1]), 2*x.shape[-1]*encScale).to(device) # interleave sins and coss
    return encX

class NerfNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depths = (3, 5) # position input is fed into the network again between these two groups of layers
        self.encScale = 6

        inputSizes = (6*self.encScale, 6*self.encScale + 256)

        self.fcLayers = nn.ModuleList()
        for depth, inputSize in zip(self.depths, inputSizes):
            layers = [nn.Linear(inputSize, 256), nn.ReLU()]
            for _ in range(depth - 1):
                layers.append(nn.Linear(256, 256))
                layers.append(nn.ReLU())

            self.fcLayers.append(nn.Sequential(*layers))
    

        self.densityModule = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.colourModule = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # we run the training loop on a batch of rays. the renderer turns this into a 2d batch, with the second dimension being the samples along each ray
        # flatten these into one dimension before passing to the model
        xShape = x.shape
        x = x.flatten(end_dim=-2) 
        encX = posEncode(x, self.encScale)

        logits = self.fcLayers[0](encX)
        logits = self.fcLayers[1](torch.cat((logits, encX), dim=-1))

        density = self.densityModule(logits)
        colour = self.colourModule(logits)

        # unflatten
        density = density.view(*(xShape[:-1]))
        colour = colour.view(*(xShape[:-1]), 3)
        return density, colour




def train(dataloader, model, optimiser, lossFn = nn.MSELoss()):
    model.train()
    for batch, (rays, trueCols) in enumerate(dataloader):
        rays, trueCols = rays.to(device), trueCols.to(device)

        rayOrigins, rayDirs = rays.split(3, dim=-1)
        rayOrigins, rayDirs = rayOrigins.to(device), rayDirs.to(device)
        predCols = renderRays(rayOrigins, rayDirs, model, device=device)

        loss = lossFn(predCols, trueCols) 

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 2000 == 0:
            size = len(dataloader.dataset)
            loss = loss.item()
            current = (batch + 1) * len(rays)
            print(f"loss: {loss:>.3f} [{current:>5d}/{size:>5d}]")


def savePredImg(model, pose, focal, path, name):
    model.eval()
    with torch.no_grad():
        img = renderScene(model, 100, 100, focal, pose, device=device)
        
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


def run():
    outpath = "output/"
    dataset = NerfDataset(device)
    dataloader = DataLoader(dataset, 10000, shuffle=True)

    model = NerfNet().to(device)
    model.load_state_dict(torch.load(outpath + "model/model.pth", weights_only=True))

    # optimiser = torch.optim.SGD(model.parameters(), lr = 5e-3, momentum=0.9) # 5e-4 is the original implementation's initial learning rate
    optimiser = torch.optim.Adam(model.parameters())
    scheduler = lr.StepLR(optimiser, step_size=500, gamma=0.5)

    epochs = 10000
    t = time()
    for epoch in range(epochs):
        print(f"Time {(time() - t):.2f}s")
        print(f"Memory use: {(torch.cuda.max_memory_allocated(device=device) // 1e6):.0f}MiB")
        t = time()

        print(f"Epoch {epoch}")
        train(dataloader, model, optimiser)
        scheduler.step()
        
        if epoch % 10 == 0:
            savePredImg(model, dataset.poses[25], dataset.focal, outpath + "img/", f"{(epoch//10):05}.png")
            saveModel(model, outpath + "model/", "model.pth")


run()