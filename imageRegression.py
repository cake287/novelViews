import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.io as io
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# input and output paths. do not include file extension on output path
images = [(R"melon.jpg", R"output\melon"), (R"monkey.png", R"output\monkey")]

def DataFromPath(path):
    image = io.read_image(path, io.ImageReadMode.RGB)
    image = image.permute(1, 2, 0)

    res = image.shape[0]

    spaced_coords = np.linspace(0, 1, res + 1)[:-1]
    norm_coords = np.stack(np.meshgrid(spaced_coords, spaced_coords), axis=-1)
    x_train = torch.from_numpy(norm_coords.reshape(-1, 2)).to(torch.float32) # a list of normalised coordinates (one for each pixel) 

    y_train = image.reshape(-1, 3).float() / 255

    train = TensorDataset(x_train, y_train)
    return (DataLoader(train, batch_size=1000, shuffle=True), x_train, res)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class NoEncNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "no-enc"
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class GaussEncNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.name = "gauss-enc"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  
            nn.Sigmoid()
        )
        self.encWeights = torch.from_numpy(np.random.normal(0, 1, size=(256, 2))).float().to(device)

    def forward(self, x):
        #x = self.flatten(x)

        # this stuff gets really annoying because x isn't a single input, it's a batch of inputs
        
        products = torch.matmul(x, self.encWeights.T)
        sins = torch.sin(2*math.pi * products)
        coss = torch.cos(2*math.pi * products)
        # encX = torch.cat((coss, sins), dim=1)
        encX = torch.stack((coss, sins), dim=2).reshape(x.size()[0], 2*self.encWeights.size()[0]) # interleave coss and sins
        logits = self.linear_relu_stack(encX)
        return logits


class PosEncNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.name = "pos-enc"
        self.encScale = 6
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*self.encScale, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  
            nn.Sigmoid()
        )
    
    def forward(self, x):
        #x = self.flatten(x)
        pows = torch.pow(2, torch.arange(0, self.encScale))
        products = x.unsqueeze(2) * pows
        products = products.view(x.size(0), 2*self.encScale)

        sins = torch.sin(2*math.pi * products)
        coss = torch.cos(2*math.pi * products)
        encX = torch.stack((coss, sins), dim=2).reshape(x.size()[0], 4*self.encScale) # interleave sins and coss 

        logits = self.linear_relu_stack(encX)
        return logits



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) 

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10000 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>5f} [{current:>5d}/{size:>5d}]")


def save_pred_image(model, res, x_train, path, name):
    model.eval()
    with torch.no_grad():
        x_train2 = x_train.to(device)

        cols = model(x_train2)
        cols = torch.clamp(cols, min=0, max=1)
        img = cols.reshape(res, res, 3)
        
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(img.permute(2, 0, 1), f"{path}{name}")
        print("saved predicted image")

        # plt.axis('off')
        # plt.imshow(img)
        # plt.show(block = False)
        # plt.show()


for (inputPath, outputPath) in images:
    (train_dataloader, x_train, res) = DataFromPath(inputPath)
    for model in [GaussEncNetwork().to(device), PosEncNetwork().to(device), NoEncNetwork().to(device)]:
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)

        epochs = 20
        for t in range(epochs):
            if t % 10 == 9:
                print(f"Epoch {t}")
                save_pred_image(model, res, x_train, f"{outputPath}/{model.name}/", f"{t // 1000}k.png")
            train(train_dataloader, model, loss_fn, optimizer)

