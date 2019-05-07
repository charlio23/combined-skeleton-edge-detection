import os
import torch
from dataset import SKLARGE_TEST
from network import CombinedHED_FSDS
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

rootDirImgTest = "../DeepSkeleton-pytorch/SK-LARGE/images/test/"
testEdgeOutput = "output-edge/"
testSkeletonOutput = "output-skeleton/"

testDS = SKLARGE_TEST(rootDirImgTest)
test = DataLoader(testDS, shuffle=False)

os.makedirs(testEdgeOutput, exist_ok=True)
os.makedirs(testSkeletonOutput, exist_ok=True)


print("Loading trained network...")

networkPath = "checkpoints/COMBINED-SKLARGE.pth"

nnet = CombinedHED_FSDS()
dic = torch.load(networkPath)
dicli = list(dic.keys())
new = {}
j = 0

for k in nnet.state_dict():
    new[k] = dic[dicli[j]]
    j += 1

nnet.load_state_dict(new)

print("Generating test results...")
soft = torch.nn.Softmax(dim=1)
for data in tqdm(test):
    image, imgName = data
    image = Variable(image, requires_grad=False)
    sideOuts = nnet(image)
    fuseEdge = grayTrans(sideOuts[5])
    fuseSkeleton = grayTrans((1 - soft(sideOuts[10])[0][0]).unsqueeze_(0).unsqueeze_(0))
    fuseEdge.save(testEdgeOutput + imgName[0])
    fuseSkeleton.save(testSkeletonOutput + imgName[0])
