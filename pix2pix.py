import argparse
from network import *
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pix2pix import Wrapper
from nonmaxsuppression import nms
from collections import OrderedDict
from torch.autograd import Variable

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img



w = Wrapper()
net_path = "checkpoints/final/Combined-COCO-pix2pix.pth"

nnet = CombinedHED_FSDS().cuda()

dic = torch.load(net_path)
dicli = list(dic.keys())
new = {}
j = 0
for k in nnet.state_dict():
    new[k] = dic[dicli[j]]
    j += 1
nnet.load_state_dict(new)


transf = transforms.ToTensor()

out_dir = "examples-mapping/"
data_path = "../val2017/"

samples = os.listdir(data_path + "images")
for halo in tqdm(range(1,2)):
    num = np.random.randint(0,len(samples))
    print(num)
    sampleName = samples[num]

    image = transf(Image.open(data_path + "images/" + sampleName).convert('RGB'))
    tensorBlue = (image[0:1, :, :] * 255.0) - 104.00698793
    tensorGreen = (image[1:2, :, :] * 255.0) - 116.66876762
    tensorRed = (image[2:3, :, :] * 255.0) - 122.67891434
    inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0).unsqueeze_(0)

    receptive_fields = np.array([14,40,92,196])
    p = 1.2

    def obtain_scale_map(loc_map, scale_map):
        batch, _, height, width = loc_map.size()
        value, ind = loc_map[0,1:].max(0)
        probability_map = 2*(1 - soft(loc_map)[:,0:1]) - 1
        scale_map = [(scale_map[i] + 1)*receptive_fields[i]/2 for i in range(0,len(scale_map))]
        scale_map = torch.cat(scale_map, 1)
        scale_map = 2*(scale_map.gather(1,ind.unsqueeze_(0).unsqueeze_(0)))/255.0 - 1
        result = torch.cat([probability_map, scale_map],1)
        return result

    sideOuts = nnet(inputImage.cuda())
    edgeOuts = sideOuts[:6]
    skeletonOuts = sideOuts[6:11]
    scaleOuts = sideOuts[11:]
    soft = torch.nn.Softmax(dim=1)

    fused_edge = edgeOuts[-1]
    fused_skeleton = (1 - soft(skeletonOuts[-1])[0][0]).unsqueeze_(0).unsqueeze_(0)
    scale_map = obtain_scale_map(skeletonOuts[-1], scaleOuts).cuda()
    edge_nms = nms(fused_edge).float().detach().cuda()
    skeleton_nms = nms(fused_skeleton).float().cuda()
    skeleton_nms[skeleton_nms < 0.5] = 0
    skeleton_nms[skeleton_nms >= 0.5] = 1
    skeleton_nms = (skeleton_nms*scale_map[:,1:]).detach()

    fused_edge = Variable(2*fused_edge - 1,requires_grad=False).cuda()
    edge_nms = Variable(2*edge_nms - 1,requires_grad=False).cuda()

    scale_map = Variable(scale_map,requires_grad=False).cuda()
    skeleton_nms = Variable(skeleton_nms,requires_grad=False).cuda()

    skeleton_nms = Variable(skeleton_nms,requires_grad=False)
    edge_input = {'A': fused_edge, 'B': skeleton_nms, 'A_paths': None}
    ske_input = {'A': scale_map, 'B': edge_nms, 'A_paths': None}

    ske_new = (w.edge_to_skeleton(edge_input) + 1)/2
    edge_new = (w.skeleton_to_edge(ske_input) + 1)/2


    #edge_new, ske_new = w.map_and_optimize(2*fused_edge.cpu() - 1, scale_map.cpu(),2*edge_nms.cpu() - 1, skeleton_nms.cpu())

    grayTrans(edge_nms).save(out_dir + "edge_ori_" + str(num) + ".png")
    grayTrans((skeleton_nms)).save(out_dir + "ske_ori_" + str(num) + ".png")
    grayTrans(edge_new).save(out_dir + "mapped_edge_" + str(num) + ".png")
    grayTrans(ske_new).save(out_dir + "mapped_ske_" + str(num) + ".png")
