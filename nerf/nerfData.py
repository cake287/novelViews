import numpy as np
import torch
from torch.utils.data import Dataset
import volumeRendering
# from volumeRendering import showImage

path = R"data/nerf/tiny_nerf_data.npz"

class NerfDataset(Dataset):
    def __init__(self, device):
        data = np.load(path)
        self.images = torch.tensor(data["images"]).to(device)
        self.poses = torch.tensor(data["poses"]).to(device)
        self.focal = 3 #data["focal"]
        
        self.imageCount = self.images.shape[0]
        self.H, self.W = self.images.shape[1:3]

        # xs, ys = torch.meshgrid(torch.linspace(-1, 1, self.W), torch.linspace(-1, 1, self.H), indexing="xy") # uv coordinates
        # uvs = torch.stack((xs, ys), dim=-1).unsqueeze(0).expand(self.imageCount, -1, -1, -1)
        # expandedPoses = self.poses.view(self.imageCount, 1, 1, 4, 4).expand(-1, self.H, self.W, -1, -1) # repeat camera matrix for each uv coord in an image
        # rayOrigins, rayDirs = volumeRendering.getRays(uvs, 3, expandedPoses)

        rayOrigins, rayDirs = volumeRendering.getRays(self.H, self.W, 3, self.poses, device)
        self.rays = torch.cat((rayOrigins, rayDirs), -1).to(device)

    def __len__(self):
        return self.imageCount * self.H * self.W
    
    def __getitem__(self, idx):
        img = idx // (self.H * self.W)
        pixelIdx = idx % (self.H * self.W)
        Y = pixelIdx // self.W
        X = pixelIdx % self.W
        
        # return self.uvs[Y, X], self.images[img, Y, X] # for testUVs
        return self.rays[img, Y, X], self.images[img, Y, X]




def testUVs():
    """tests __getitem__ returning uv coords"""
    ds = NerfDataset()
    
    testImg = torch.zeros(ds.H, ds.W, 3)
    imgIdx = 50
    for y in range(ds.H):
        for x in range(ds.W):
            uv, col = ds.__getitem__(imgIdx*ds.H*ds.W + y*ds.H + x)
            if (torch.norm(uv) > 1):
                col = torch.tensor([1, 0, 0])
            
            testImg[y, x] = col

    # showImage(testImg)
