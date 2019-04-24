import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import pandas as pd

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