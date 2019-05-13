import os
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from tqdm import tqdm
from matplotlib import pyplot as plt

rootDir = "../train2017/groundTruth"

edgeDir = rootDir + "/edges/"
skeletonDir = rootDir + "/skeletons/"
scaleDir = rootDir + "/scale/"
os.makedirs(scaleDir, exist_ok=True)

fileList = os.listdir(edgeDir)

for fileName in tqdm(fileList):
    edge = (np.array(Image.open(edgeDir + fileName).convert('L')) > 200).astype(float)
    skeleton = (np.array(Image.open(skeletonDir + fileName).convert('L')) > 200).astype(float)
    dist = 2.0*bwdist(1.0 - edge)
    make_scale = np.vectorize(lambda x, y: 0 if y < 0.99 else x)

    scale = make_scale(dist, skeleton).astype(np.uint8)
    edge = (edge*255.0).astype(np.uint8)
    skeleton = (skeleton*255.0).astype(np.uint8)
    fileName = fileName.replace('.jpg','.png')
    
    Image.fromarray(edge, 'L').save(edgeDir + fileName)
    Image.fromarray(skeleton, 'L').save(skeletonDir + fileName)
    Image.fromarray(scale, 'L').save(scaleDir + fileName)
