import torch as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import randint

def showImage(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show(block = False)
    plt.show()
def plotPoints(points, opacities = None, points2 = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (coords, col) in ([(points, 'b')] + ([(points2, 'g')] if points2 != None else [])):
        coords = coords.reshape(-1, 3)
        xs = coords[:, 0].numpy()
        ys = coords[:, 1].numpy()
        zs = coords[:, 2].numpy()

        if opacities == None:
            ax.scatter(xs, ys, zs, color=col, marker='o')
        else:
            opacities = opacities.reshape(-1)
            # t.set_printoptions(profile="full")
            # print(opacities)
            colours = np.column_stack((np.tile([0.5, 0, 1], (len(xs), 1)), opacities))
            ax.scatter(xs, ys, zs, c=colours, marker='o')

    ax.scatter([0], [0], [0], color='r', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def getRays(H, W, focal, c2w):
    """
    Generate H*W rays for each camera matrix in c2w

    focal is a focal length for each camera matrix - CURRENTLY FOCAL IS JUST ONE VALUE
    """
    # generate uv coords (-1 to 1)
    xs, ys = t.meshgrid(t.linspace(-1, 1, W), t.linspace(-1, 1, H), indexing="xy")

    # expand xs and ys for each matrix in c2w (agnostic to c2w's shape, oh yeah. unreadable but sick)
    preDims = len(c2w.shape) - 2
    xs = xs.view(*([1] * preDims), *xs.shape).expand(*c2w.shape[:-2], -1, -1)
    ys = ys.view(*([1] * preDims), *ys.shape).expand(*c2w.shape[:-2], -1, -1)

    rayDirs = t.stack([xs, -ys, -focal * t.ones_like(xs)], -1) # dirs in camera space
    rayDirs /= t.norm(rayDirs, dim=-1, keepdim=True) # normalise

    # rotate by camera matrix. presumably there is a better way than all this squeezing. who knows
    rayDirs = t.matmul(c2w[..., :3, :3].unsqueeze(-3).unsqueeze(-3), rayDirs.unsqueeze(-1)).squeeze(-1) 
    
    rayOrigins = c2w[..., :3, 3].view(*(c2w.shape[:-2]), 1, 1, 3).expand_as(rayDirs)
    return rayOrigins, rayDirs


# def getRays(uv, focal: float, c2w):
#     """
#     Generate a ray for each uv coordinate (x and y coords from -1 to 1), given their corresponding camera parameters.

#     focal is a focal length for each uv coordinate - CURRENTLY FOCAL IS JUST ONE VALUE

#     c2w is a camara matrix for each uv coordinate
#     """
#     xs, ys = uv[..., 0], uv[..., 1]

#     rayDirs = t.stack([xs, -ys, -focal * t.ones_like(xs)], -1) # dirs in camera space
#     rayDirs /= t.norm(rayDirs, dim=-1, keepdim=True) # normalise

#     rayDirs = t.matmul(c2w[..., :3, :3], rayDirs.unsqueeze(-1)).squeeze(-1) # rotate by camera matrix
    
#     rayOrigins = c2w[..., :3, 3]
#     return rayOrigins, rayDirs

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

def renderRays(rayOrigins, rayDirs, sceneFunc, sampleCount=120, near=0, far=8, randSamples=True):
    """Renders the given batch of rays and returns their colours"""

    # choose sampling points for each ray. atm these are at regular intervals - TODO change to random
    # there are (n+1) sample depths so that the nth sample has an interval distance for the integration approximation. this (n+1)th depth is not actually sampled.
    sampleDepths = t.linspace(near, far, sampleCount + 1)
    sampleDepths = sampleDepths.expand(rayOrigins.shape[:-1] + sampleDepths.shape)
    # sampleDepthsRegular = sampleDepths.clone()

    if (randSamples):
        sampleDepths = sampleDepths + ((far - near) / sampleCount) * t.rand(sampleDepths.shape) # move each depth forward to by a random distance within its bin
    samplePoints = rayOrigins.unsqueeze(-2) + sampleDepths[..., :-1].unsqueeze(-1) * rayDirs.unsqueeze(-2)

    # samplePointsRegular = rayOrigins.unsqueeze(-2) + sampleDepthsRegular.unsqueeze(-1) * rayDirs.unsqueeze(-2)

    # test sample point generation
    if (False):
        plotPoints(samplePointsRegular)#, points2 = samplePointsRegular)#, ellipsoidDensity(samplePoints).clamp(0.05, 1))


    sceneDensities, sceneSourceColours = sceneFunc(samplePoints)
    sampleDiffs = sampleDepths.diff()

    # the probabilities that the ray has got to that point without hitting an object and stopping
    accumulatedTransmittances = t.exp(-t.cumsum(sceneDensities*sampleDiffs, dim=-1))

    # the colour of the scene at this point (=density*colour)
    sceneColours = sceneSourceColours * (1 - t.exp(-sceneDensities*sampleDiffs)).unsqueeze(-1)

    pixelColours = t.sum(sceneColours * accumulatedTransmittances.unsqueeze(-1), dim=-2)
    return pixelColours


def renderScene(sceneFunc, W, H, focal, c2w, sampleCount=120, near=0, far=8):
    """Renders the given scene function as a W*H image"""
    # xs, ys = t.meshgrid(t.linspace(-1, 1, W), t.linspace(-1, 1, H), indexing="xy") # uv coordinates
    # c2ws = c2w.unsqueeze(0).unsqueeze(0).expand(H, W, c2w.shape[0], c2w.shape[1])

    # rayOrigins, rayDirs = getRays(t.stack((xs, ys), dim=-1), focal, c2ws)

    rayOrigins, rayDirs = getRays(H, W, focal, c2w)
    return renderRays(rayOrigins, rayDirs, sceneFunc, sampleCount, near, far, True)


def ellipsoidDensity(pos):
    """= 1 - (|x, 0.3y, z|)^5"""
    lengths = t.norm(pos * t.tensor([1, 0.3, 1]), dim=-1)
    density = 1 - t.pow(lengths, 5)

    col = (200 * pos).clamp(0, 1)
    
    return density.clamp(0, 1), col


def test():


    data = np.load(R"data\nerf\tiny_nerf_data.npz")
    poses = t.tensor(data["poses"])
    images = t.tensor(data["images"])
    focal = data["focal"]

    

    # verify that all poses are looking in the same direction
    # this is necessary since I haven't calibrated focal length which will be needed for freeform camera poses 
    if (False):
        c2ws = poses[:30] #t.stack((poses[3], poses[6]))
        uvs = t.zeros(c2ws.shape[0], 2)
        rayOrigins, rayDirs = getRays(uvs, 2, c2ws)
        renderRays(rayOrigins, rayDirs, ellipsoidDensity, sampleCount=20)


    # c2w = t.tensor([
    #     [-0.687, -0.309, 0.658, 2.654],
    #     [0.727, -0.291, 0.621, 2.506],
    #     [0.000, 0.905, 0.424, 1.711],
    #     [0.000, 0.000, 0.000, 1.000]
    # ])
    # image = renderScene(ellipsoidDensity, 300, 300, 2, c2w, sampleCount=100)
    # showImage(image)




    imageCount = images.shape[0]

    finalImg = None

    for n in range(3):
        i = randint(0, imageCount-1)

        predImage = renderScene(ellipsoidDensity, 100, 100, 1.5, poses[i], sampleCount=100)
        # predImage = predImage.unsqueeze(-1).repeat(1, 1, 3) # turn greyscale to rgb

        adjacentImages = t.cat((images[i], predImage), dim=1)

        if finalImg == None:
            finalImg = adjacentImages
        else:
            finalImg = t.cat((finalImg, adjacentImages))

    showImage(finalImg)
