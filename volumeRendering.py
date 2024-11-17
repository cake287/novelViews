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
        # print(opacities)
        rgba_colors = np.column_stack((np.tile([0.5, 0, 1], (len(xs), 1)), opacities))
        ax.scatter(xs, ys, zs, c=rgba_colors, marker='o')

    ax.scatter([0], [0], [0], color='r', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    plt.show()

def getRays(uv, focal, c2w):
    """
    Generate a ray for each uv coordinate, given their corresponding camera parameters.
    This is intended for rays from different cameras so that a model can be trained on random rays, as opposed to training on all the rays from one image.

    focal is a focal length for each coordinate
    c2w is a camara matrix  
    """
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
# test ray generation
if (False):
    c2w = t.tensor([
        [-0.687, -0.309, 0.658, 2.654],
        [0.727, -0.291, 0.621, 2.506],
        [0.000, 0.905, 0.424, 1.711],
        [0.000, 0.000, 0.000, 1.000]
    ])
    rayOrigins, rayDirs = getRays(5, 5, 1, c2w)
    
    points = rayOrigins + rayDirs
    points = t.cat([points.reshape(-1, 3), t.unsqueeze(c2w[:3, 3], 0)], dim=0)
    plotPoints(points)

def renderRays(rayOrigins, rayDirs, sceneDensityFunc, sampleCount=6, near=0, far=8):
    """Renders the given batch of rays and returns their colours"""

    # choose sampling points for each ray. atm these are at regular intervals - TODO change to random
    # there are (n+1) sample depths so that the nth sample has an interval distance for the integration approximation. this (n+1)th depth is not actually sampled.
    sampleDepths = t.linspace(near, far, sampleCount + 1)
    sampleDepths = sampleDepths.unsqueeze(0).unsqueeze(0).expand(rayOrigins.shape[:-1] + sampleDepths.shape)
    
    samplePoints = rayOrigins.unsqueeze(-2) + sampleDepths[..., :-1].unsqueeze(-1) * rayDirs.unsqueeze(-2)

    # test sample point generation
    if (False):
        plotPoints(samplePoints, ellipsoidDensity(samplePoints).clamp(0.05, 1))

    sceneDensities = sceneDensityFunc(samplePoints)
    sampleDiffs = sampleDepths.diff()

    # the probabilities that the ray has got to that point without hitting an object and stopping
    accumulatedTransmittances = t.exp(-t.cumsum(sceneDensities*sampleDiffs, dim=-1))

    # the colour of the scene at this point (=density*colour). ATM THERE IS NO COLOUR SO THIS IS JUST DENSITY
    sceneColours = 1 - t.exp(-sceneDensities*sampleDiffs)

    pixelColours = t.sum(accumulatedTransmittances*sceneColours, dim=-1)
    return pixelColours


def renderScene(sceneDensityFunc, W, H, focal, c2w, sampleCount=6, near=0, far=8):
    """Renders the given scene function as a W*H image"""
    rayOrigins, rayDirs = getRays(H, W, focal, c2w)
    return renderRays(rayOrigins, rayDirs, sceneDensityFunc, sampleCount, near, far)


# camera to world coords transform
c2w = t.tensor([
    [-0.687, -0.309, 0.658, 2.654],
    [0.727, -0.291, 0.621, 2.506],
    [0.000, 0.905, 0.424, 1.711],
    [0.000, 0.000, 0.000, 1.000]
])


def ellipsoidDensity(pos):
    """= 1 - (|x, 0.3y, z|)^5"""
    pos = pos.clone() # don't modify the input
    pos[..., 1] *= 0.3
    lengths = t.norm(pos, dim=-1)
    density = 1 - t.pow(lengths, 5)
    return density.clamp(0, 1)

image = renderScene(ellipsoidDensity, 300, 300, 2, c2w, sampleCount=100)

plt.axis('off')
plt.imshow(image)
plt.show(block = False)
plt.show()