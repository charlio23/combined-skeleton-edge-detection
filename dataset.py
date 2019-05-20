import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt as bwdist


class COCO(Dataset):
    def __init__(self, rootDir, offline=False):
        self.rootDirImg = rootDir + "images/"
        self.rootDirGt = rootDir + "groundTruth/" + "person/" + "skeletons/"
        self.rootDirGtEdges = rootDir + "groundTruth/" + "person/" + "edges/"
        self.listData = sorted(os.listdir(self.rootDirGt))
    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i].replace('.png','.jpg')
        targetName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))

        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)

        targetImageSkeleton = transf(Image.open(self.rootDirGt + targetName).convert('L'))*255.0
        targetImageEdge = transf(Image.open(self.rootDirGtEdges + targetName).convert('L'))
        #edge = transf(Image.open(self.rootDirGtEdges + targetName).convert('L')).squeeze_(0).numpy()> 0.5
        #dist = 2.0*bwdist(1.0 - (edge.astype(float)))
        #make_scale = np.vectorize(lambda x, y: 0 if y < 0.99 else x)

        #scale = make_scale(dist,targetImage)
        #targetImage = torch.from_numpy(scale).float().unsqueeze_(0)
        return inputImage, targetImageEdge, targetImageSkeleton

class SKLARGE(Dataset):
    def __init__(self, fileNames, rootDir):
        self.rootDir = rootDir
        self.transform = transforms.ToTensor()
        self.targetTransform = transforms.ToTensor()
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        edgeTargetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1]).replace("gt_scale", "ed_scale")
        skeletonTargetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])
        # process the images
        inputImage = Image.open(inputName).convert('RGB')
        inputImage = self.transform(inputImage)

        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434

        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)

        edgeTargetImage = Image.open(edgeTargetName).convert('L')
        edgeTargetImage = self.targetTransform(edgeTargetImage)

        skeletonTargetImage = Image.open(skeletonTargetName).convert('L')
        skeletonTargetImage = self.targetTransform(skeletonTargetImage)
        
        return inputImage, edgeTargetImage, skeletonTargetImage

class SKLARGE_TEST(Dataset):
    def __init__(self, rootDirImg):
        self.rootDirImg = rootDirImg
        self.listData = sorted(os.listdir(rootDirImg))

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)
        
        inputName = inputName.split(".jpg")[0] + ".png"
        return inputImage, inputName