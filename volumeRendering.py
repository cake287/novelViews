import torch as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plotPoints(coords, opacities = None):
    coords = coords.reshape(-1, 3)
    xs = coords[:, 0].numpy()
    ys = coords[:, 1].numpy()
    zs = coords[:, 2].numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    if opacities == None:
        ax.scatter(xs, ys, zs, color='b', marker='o')
    else:
        opacities = opacities.reshape(-1)
        # t.set_printoptions(profile="full")
        print(opacities)
        rgba_colors = np.column_stack((np.tile([0.5, 0, 1], (len(xs), 1)), opacities))
        ax.scatter(xs, ys, zs, c=rgba_colors, marker='o')


    ax.scatter([0], [0], [0], color='r', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def getRays(H, W, focal, c2w):
    """
    Generate a ray for each pixel in a W*H image.
    c2w is camera matrix (4x3).

    Returns (rayOrigins, rayDirs). Both are (H, W, 3)
    """
    xs, ys = t.meshgrid(t.linspace(-1, 1, W), t.linspace(-1, 1, H), indexing="xy") # uv coordinates

    rayDirs = t.stack([xs, -ys, -focal * t.ones_like(xs)], -1) # dirs in camera space
    rayDirs /= t.norm(rayDirs, dim=-1, keepdim=True) # normalise
    rayDirs @= c2w[:3, :3].T # rotate by camera matrix

    rayOrigins = c2w[:3, 3].unsqueeze(0).unsqueeze(0).expand(H, W, -1) # repeat the origin for every dir

    return rayOrigins, rayDirs



def sampleScene(pos):
    """Sample scene for a single input or batch of inputs. Returns the scene density (i.e. probability an object exists) for these points"""
    pos[..., 1] *= 0.3
    lengths = t.norm(pos, dim=-1)
    density = 1 - t.pow(lengths, 5)
    return density.clamp(0, 1)


c2w = t.tensor([
    [-0.687, -0.309, 0.658, 2.654],
    [0.727, -0.291, 0.621, 2.506],
    [0.000, 0.905, 0.424, 1.711],
    [0.000, 0.000, 0.000, 1.000]
])
# c2w = t.tensor([
#     [1, 0, 0, 3],
#     [0, 1, 0, 1],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ], dtype=t.float)
pos = c2w[:3, 3]


# test ray generation
if (False):
    rayOrigins, rayDirs = getRays(20, 20, 1, c2w)
    
    points = rayOrigins + rayDirs
    points = t.cat([points.reshape(-1, 3), t.unsqueeze(pos, 0)], dim=0)
    plotPoints(points)



W, H = 10,10
rayOrigins, rayDirs = getRays(H, W, 1, c2w)
rayCount = rayOrigins.shape[0]

sampleCount = 15
near = 0
far = 8

# choose sampling points for each ray. atm these are at regular intervals - TODO change to random
sampleDepths = t.linspace(near, far, sampleCount).unsqueeze(0).unsqueeze(0).expand(H, W, -1)
samplePoints = rayOrigins.unsqueeze(-2) + sampleDepths.unsqueeze(-1) * rayDirs.unsqueeze(-2)

# test sample point generation
if (True):
    plotPoints(samplePoints, sampleScene(samplePoints).clamp(0.04, 1))


