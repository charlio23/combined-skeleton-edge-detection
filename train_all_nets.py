from torch.utils.data import DataLoader
import torch
from network import CombinedHED_FSDS
from dataset import SKLARGE, COCO
from pix2pix import Wrapper
from nonmaxsuppression import nms
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.nn.functional import binary_cross_entropy
from torch.nn.functional import cross_entropy, mse_loss
from torch.autograd import Variable
import time
from tqdm import tqdm
from torch.optim import lr_scheduler
from collections import defaultdict
import os
import matplotlib.pyplot as plt


jointNetworkPath = "./checkpoints/Combined-COCO.pth"
outDir = "./checkpoints/final_exp/"
network_name = "Combined-COCO-pix2pix.pth"
edge_net_D = "edge_net_D.pth"
edge_net_G = "edge_net_G.pth"
skeleton_net_D = "skeleton_net_D.pth"
skeleton_net_G = "skeleton_net_G.pth"
image_dir = "images-coco-final"
def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Loading dataset...")

trainDS = COCO("../train2017/",True)
train = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=1)
os.makedirs(outDir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

print("Loading joint edge/skeleton detectors...")

nnet = torch.nn.DataParallel(CombinedHED_FSDS()).cuda()
dic = torch.load(jointNetworkPath)
dicli = list(dic.keys())
new = {}
j = 0

for k in nnet.state_dict():
    new[k] = dic[dicli[j]]
    j += 1

nnet.load_state_dict(new)


print("Loading pix2pix mapping functions...")

w = Wrapper()

print("Defining util functions and parameters...")

### HYPER-PARAMETERS
learningRate = 1e-7
momentum = 0.9
weightDecay = 0.0002
receptive_fields = np.array([14,40,92,196])
p = 1.2
soft = torch.nn.Softmax(dim=1)

###

def balanced_binary_cross_entropy(input, target):
    batch, _, width, height = target.size()
    pos_index = (target >=0.5)
    neg_index = (target <0.5)        
    weight = torch.zeros_like(target)
    sum_num = width*height
    pos_num = pos_index.sum().cpu().item()
    neg_num = sum_num - pos_num
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num / sum_num
    weight[neg_index] = pos_num / sum_num 
    loss = binary_cross_entropy(input, target.float(), weight.float(), reduction='none')

    return torch.sum(loss)/batch

def balanced_cross_entropy(input, target):
    batch, _, height, width = target.size()
    total_weight = height*width
    pos_weight = torch.sum(target > 0.1).item()/total_weight
    neg_weight = 1 - pos_weight
    weights = torch.ones(input.size(1))*neg_weight
    weights[0] = pos_weight
    #CE loss
    loss = cross_entropy(input,target.squeeze_(1),weight=weights.cuda(),reduction='none')

    return torch.sum(loss)/batch

def regressor_loss(input, targetScale, targetQuant):
    weight = (targetQuant > 0.01).float()
    loss = torch.sum(weight*mse_loss(input, targetScale, reduction='none'))
    batch = targetScale.shape[0]
    return loss/batch

def generate_quantise(quantise):
    result = []
    for i in range(1,5):
        result.append(quantise*(quantise <= i).long())

    result.append(quantise)
    return result

def generate_scales(quant_list, fields, scale):
    result = []
    for quantise, r in zip(quant_list,fields):
        result.append(2*((quantise > 0).float()*scale)/r - 1)
    return result

def apply_quantization(scale):
    if scale < 0.001:
        return 0
    if p*scale > np.max(receptive_fields):
        return len(receptive_fields)
    return np.argmax(receptive_fields > p*scale) + 1

def obtain_scale_map(loc_map, scale_map):
    batch, _, height, width = loc_map.size()
    value, ind = loc_map[0,1:].max(0)
    probability_map = 2*(1 - soft(loc_map)[:,0:1]) - 1
    scale_map = [(scale_map[i] + 1)*receptive_fields[i]/2 for i in range(0,len(scale_map))]
    scale_map = torch.cat(scale_map, 1)
    scale_map = 2*(scale_map.gather(1,ind.unsqueeze_(0).unsqueeze_(0)))/255.0 - 1
    result = torch.cat([probability_map, scale_map],1)
    return result

print("Defining optimizer for combined edge/skeleton network...")

# Optimizer settings.
net_parameters_id = defaultdict(list)
for name, param in nnet.named_parameters():
    if name in ['module.conv1.0.weight', 'module.conv1.2.weight',
                'module.conv2.1.weight', 'module.conv2.3.weight',
                'module.conv3.1.weight', 'module.conv3.3.weight',
                'module.conv3.5.weight', 'module.conv4.1.weight',
                'module.conv4.3.weight', 'module.conv4.5.weight']:
        print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id['conv1-4.weight'].append(param)
    elif name in ['module.conv1.0.bias', 'module.conv1.2.bias',
                  'module.conv2.1.bias', 'module.conv2.3.bias',
                  'module.conv3.1.bias', 'module.conv3.3.bias',
                  'module.conv3.5.bias', 'module.conv4.1.bias',
                  'module.conv4.3.bias', 'module.conv4.5.bias']:
        print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id['conv1-4.bias'].append(param)
    elif name in ['module.conv5.1.weight', 'module.conv5.3.weight',
                  'module.conv5.5.weight']:
        print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id['conv5.weight'].append(param)
    elif name in ['module.conv5.1.bias', 'module.conv5.3.bias',
                  'module.conv5.5.bias']:
        print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id['conv5.bias'].append(param)
    elif name in ['module.edgeSideOut1.weight', 'module.edgeSideOut2.weight',
                  'module.edgeSideOut3.weight',
                  'module.edgeSideOut4.weight', 'module.edgeSideOut5.weight','module.skeletonFuseScale2.weight', 
                  'module.skeletonSideOut2.weight','module.skeletonSideOut3.weight',
                  'module.skeletonSideOut4.weight','module.skeletonSideOut5.weight',
                  'module.skeletonSideOutScale2.weight', 'module.skeletonSideOutScale3.weight',
                  'module.skeletonSideOutScale3.weight', 'module.skeletonSideOutScale5.weight']:
        print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
    elif name in ['module.edgeSideOut1.bias', 'module.edgeSideOut2.bias',
                  'module.edgeSideOut3.bias', 'module.edgeSideOut4.bias',
                  'module.edgeSideOut5.bias',
                  'module.skeletonSideOut2.bias','module.skeletonSideOut3.bias',
                  'module.skeletonSideOut4.bias','module.skeletonSideOut5.bias'
                  'module.skeletonSideOutScale2.bias', 'module.skeletonSideOutScale3.bias',
                  'module.skeletonSideOutScale3.bias', 'module.skeletonSideOutScale5.bias']:
        print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id['score_dsn_1-5.bias'].append(param)
    elif name in ['module.skeletonFuseScale0.weight', 'module.skeletonFuseScale1.weight',
                  'module.skeletonFuseScale3.weight','module.skeletonFuseScale4.weight', 'module.edgeFuse.weight']:
        print('{:26} lr:0.05 decay:1'.format(name)); net_parameters_id['score_final.weight'].append(param)
    elif name in ['module.skeletonFuseScale0.bias', 'module.skeletonFuseScale1.bias',
                  'module.skeletonFuseScale2.bias', 'module.skeletonFuseScale3.bias',
                  'module.skeletonFuseScale4.bias', 'module.edgeFuse.bias']:
        print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id['score_final.bias'].append(param)

# Create optimizer.
optimizer = torch.optim.SGD([
    {'params': net_parameters_id['conv1-4.weight']      , 'lr': learningRate*1    , 'weight_decay': weightDecay},
    {'params': net_parameters_id['conv1-4.bias']        , 'lr': learningRate*2    , 'weight_decay': 0.},
    {'params': net_parameters_id['conv5.weight']        , 'lr': learningRate*100  , 'weight_decay': weightDecay},
    {'params': net_parameters_id['conv5.bias']          , 'lr': learningRate*200  , 'weight_decay': 0.},
    {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': learningRate*0.01 , 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': learningRate*0.02 , 'weight_decay': 0.},
    {'params': net_parameters_id['score_final.weight']  , 'lr': learningRate*0.05, 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_final.bias']    , 'lr': learningRate*0.002, 'weight_decay': 0.},
], lr=learningRate, momentum=momentum, weight_decay=weightDecay)

# Learning rate scheduler.
lr_schd = lr_scheduler.StepLR(optimizer, step_size=3e4, gamma=0.1)

print("Training started!")
epochs = 40
i = 1
dispInterval = 500
lossAcc = 0.0
train_size = 4
epoch_line = []
loss_line = []
nnet.train()
L = 0.5
optimizer.zero_grad()

for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    for j, (image, edge, skeleton) in enumerate(tqdm(train), 1):
        quantization = np.vectorize(apply_quantization)
        quantise = torch.from_numpy(quantization(skeleton.numpy())).cuda()
        quant_list = generate_quantise(quantise)

        image = Variable(image).cuda()
        edge = Variable(edge).cuda()
        skeleton = Variable(skeleton).cuda()

        scale_list = generate_scales(quant_list, receptive_fields, skeleton)

        sideOuts = nnet(image)
        edgeOuts = sideOuts[:6]
        skeletonOuts = sideOuts[6:11]
        scaleOuts = sideOuts[11:]

        #original loss from combined network

        loss_edge = sum([balanced_binary_cross_entropy(sideOut, edge) for sideOut in edgeOuts])
        loss_skeleton = sum([balanced_cross_entropy(sideOut, quant) for sideOut, quant in zip(skeletonOuts,quant_list)])
        loss_list_scale = sum([regressor_loss(sideOut, scale, quant) for sideOut, scale, quant in zip(scaleOuts,scale_list,quant_list[0:4])])
        
        #pix2pix loss
        fused_edge = edgeOuts[-1]
        fused_skeleton = (1 - soft(skeletonOuts[-1])[0][0]).unsqueeze_(0).unsqueeze_(0)
        scale_map = obtain_scale_map(skeletonOuts[-1], scaleOuts).cuda()
        edge_nms = nms(fused_edge).float().detach().cuda()
        skeleton_nms = nms(fused_skeleton).float().cuda()
        skeleton_nms[skeleton_nms < 0.5] = 0
        skeleton_nms[skeleton_nms >= 0.5] = 1
        skeleton_nms = (skeleton_nms*scale_map[:,1:]).detach()

        edge_new, ske_new = w.map_and_optimize(2*fused_edge - 1, scale_map,2*edge_nms - 1, skeleton_nms)
               
        loss_pix2pix_edge = balanced_binary_cross_entropy((edge_new + 1)/2, edge_nms)
        loss_pix2pix_skeleton = regressor_loss(ske_new, skeleton_nms, skeleton_nms)

        loss = loss_edge + loss_skeleton + L*loss_list_scale + loss_pix2pix_skeleton
        lossAvg = loss/train_size
        lossAvg.backward()

        if j % train_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_schd.step()
            
        lossAcc += loss.clone().item()

        if j%train_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_schd.step()
        if i%dispInterval == 0:
            timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            lossDisp = lossAcc/dispInterval
            epoch_line.append(epoch + (j - 1)/len(train))
            loss_line.append(lossDisp)
            print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i, lossDisp))
            lossAcc = 0.0
            torch.save(nnet.state_dict(), outDir + network_name)
            w.save_models(outDir, edge_net_D, edge_net_G, skeleton_net_D, skeleton_net_G)
            plt.clf()
            plt.plot(epoch_line,loss_line)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(image_dir + "/loss.png")
            plt.clf()
            fig = plt.figure(figsize=(15,5))
            for k in range(0,6):
                plt.subplot(1,7,k + 1)
                sideImg = grayTrans(sideOuts[k])
                plt.imshow(sideImg)
            plt.subplot(1,7,7)
            sideImg = grayTrans(edge)
            plt.imshow(sideImg)
            plt.savefig(image_dir + "/edge_detection.png")
            plt.clf()
            fig = plt.figure(figsize=(15,5))
            for k in range(6,11):
                plt.subplot(1,6,k - 5)
                sideImg = grayTrans((1 - soft(sideOuts[k])[0][0]).unsqueeze_(0).unsqueeze_(0))
                plt.imshow(sideImg)
            plt.subplot(1,6,6)
            sideImg = grayTrans((quantise > 0.5).unsqueeze_(0))
            plt.imshow(sideImg)
            plt.savefig(image_dir + "/skeleton_detection.png")
            plt.clf()
            fig = plt.figure(figsize=(15,5))
            plt.subplot(2,2,1)
            plt.imshow(grayTrans(edgeOuts[-1]))
            plt.subplot(2,2,2)
            plt.imshow(grayTrans((ske_new + 1)/2))
            plt.subplot(2,2,3)
            plt.imshow(grayTrans(fused_skeleton))
            plt.subplot(2,2,4)
            plt.imshow(grayTrans((edge_new + 1)/2))
            plt.savefig(image_dir + "/mapping.png")
            plt.clf()
        i += 1
