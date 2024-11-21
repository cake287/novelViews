import torch
import PIL
from volumeRendering import renderScene #, getRays
import numpy as np
from nerf import *
from torchvision.transforms.functional import to_pil_image
from math import sin, cos

# find mean and sd of camera distances in dataset
if (False):
    data = np.load(R"data\nerf\tiny_nerf_data.npz")
    poses = t.tensor(data["poses"])
    images = t.tensor(data["images"])
    focal = 3 #data["focal"]

    rayOrigins, _ = getRays(1, 1, 3, poses)
    dists = torch.norm(rayOrigins, dim=-1).view(poses.shape[0])
    mean = dists.mean()
    sd = dists.std()
    print(mean, sd)


model = NerfNet().to(device)
model.load_state_dict(torch.load("results/model.pth", weights_only=True, map_location=torch.device(device)))

data = np.load(R"data\nerf\tiny_nerf_data.npz")
poses = torch.tensor(data["poses"])

outpath = "output/gif/orbit.gif"

def normalise(v):
    return v / torch.norm(v)

def camMat(yaw):
    camPos = torch.tensor([3.5*cos(yaw), 3.5*sin(yaw), 2])
    print(camPos)
    ww = normalise(-camPos)
    print(ww)
    uu = normalise(torch.cross(ww, torch.tensor([0, 0, 1])))
    vv = normalise(torch.cross(uu, ww))
    mat3 = torch.stack((uu, vv, ww), dim=1)

    print(mat3)

camMat(0)


def renderOrbit(sceneFunc, duration=2000, steps=5, W=50, H=50, focal=3):
    imgs = []
    for step in range(steps):
        
        img = renderScene(sceneFunc, W, H, focal, poses[step], device=device)
        img = img.permute(2, 0, 1)
        imgs.append(to_pil_image(img))

        print(step, poses[step])

    
    print(imgs[1:])

    imgs[0].save(outpath, format="GIF", append_images=imgs[1:], save_all=True, duration=duration/steps, loop=0, optimize=False)



# renderOrbit(model)
