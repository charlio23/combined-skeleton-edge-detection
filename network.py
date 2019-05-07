import torch 
from torch.nn.functional import interpolate
from torch import sigmoid
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.in_channels < 6:
            n = float(m.in_channels)
            torch.nn.init.constant_(m.weight.data,1/n)
        else:
            torch.nn.init.constant_(m.weight.data,0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)

def load_vgg16(net, path):
    vgg16_items = list(torch.load(path).items())
    net.apply(weights_init)
    j = 0
    for k, v in net.state_dict().items():
        if k.find("conv") != -1:
            net.state_dict()[k].copy_(vgg16_items[j][1])
            j += 1
    return net

def initialize_net(path):
    net = CombinedHED_FSDS()
    return load_vgg16(net,path)

class CombinedHED_FSDS(torch.nn.Module):
    def __init__(self):
        super(CombinedHED_FSDS, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=35),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        # Edge Detection

        self.edgeSideOut1 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.edgeSideOut2 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.edgeSideOut3 = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.edgeSideOut4 = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.edgeSideOut5 = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.edgeFuse = torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Skeleton Localization

        self.skeletonSideOut2 = torch.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0)

        self.skeletonSideOut3 = torch.nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, padding=0)

        self.skeletonSideOut4 = torch.nn.Conv2d(in_channels=512, out_channels=4, kernel_size=1, stride=1, padding=0)

        self.skeletonSideOut5 = torch.nn.Conv2d(in_channels=512, out_channels=5, kernel_size=1, stride=1, padding=0)

        self.skeletonFuseScale0 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.skeletonFuseScale1 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.skeletonFuseScale2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.skeletonFuseScale3 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.skeletonFuseScale4 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Bilinear weights for upsampling

        self.edgeWeightDeconv2 = make_bilinear_weights(4, 1)
        self.edgeWeightDeconv3 = make_bilinear_weights(8, 1)
        self.edgeWeightDeconv4 = make_bilinear_weights(16, 1)
        self.edgeWeightDeconv5 = make_bilinear_weights(32, 1)

        self.skeletonWeightDeconv2 = make_bilinear_weights(4, 2)
        self.skeletonWeightDeconv3 = make_bilinear_weights(8, 3)
        self.skeletonWeightDeconv4 = make_bilinear_weights(16, 4)
        self.skeletonWeightDeconv5 = make_bilinear_weights(32, 5)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()
    
    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, image):

        height = image.size(2)
        width = image.size(3)

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Edge detection step

        edgeSideOut1 = self.edgeSideOut1(conv1)
        edgeSideOut2 = self.edgeSideOut2(conv2)
        edgeSideOut3 = self.edgeSideOut3(conv3)
        edgeSideOut4 = self.edgeSideOut4(conv4)
        edgeSideOut5 = self.edgeSideOut5(conv5)

        edgeUpsample2 = torch.nn.functional.conv_transpose2d(edgeSideOut2, self.edgeWeightDeconv2, stride=2)
        edgeUpsample3 = torch.nn.functional.conv_transpose2d(edgeSideOut3, self.edgeWeightDeconv3, stride=4)
        edgeUpsample4 = torch.nn.functional.conv_transpose2d(edgeSideOut4, self.edgeWeightDeconv4, stride=8)
        edgeUpsample5 = torch.nn.functional.conv_transpose2d(edgeSideOut5, self.edgeWeightDeconv5, stride=16)

                # Aligned cropping.
        edgeSideOut1 = edgeSideOut1[:, :, self.crop1_margin:self.crop1_margin+height, self.crop1_margin:self.crop1_margin+width]
        edgeSideOut2 = edgeUpsample2[:, :, self.crop2_margin:self.crop2_margin+height, self.crop2_margin:self.crop2_margin+width]
        edgeSideOut3 = edgeUpsample3[:, :, self.crop3_margin:self.crop3_margin+height, self.crop3_margin:self.crop3_margin+width]
        edgeSideOut4 = edgeUpsample4[:, :, self.crop4_margin:self.crop4_margin+height, self.crop4_margin:self.crop4_margin+width]
        edgeSideOut5 = edgeUpsample5[:, :, self.crop5_margin:self.crop5_margin+height, self.crop5_margin:self.crop5_margin+width]

        edgeFuse = self.edgeFuse(torch.cat((edgeSideOut1, edgeSideOut2, edgeSideOut3, edgeSideOut4, edgeSideOut5), 1))

        edgeSideOut1 = sigmoid(edgeSideOut1)
        edgeSideOut2 = sigmoid(edgeSideOut2)
        edgeSideOut3 = sigmoid(edgeSideOut3)
        edgeSideOut4 = sigmoid(edgeSideOut4)
        edgeSideOut5 = sigmoid(edgeSideOut5)
        edgeFuse = sigmoid(edgeFuse)

        # Skeleton detection step

        skeletonSideOut2 = self.skeletonSideOut2(conv2)
        skeletonSideOut3 = self.skeletonSideOut3(conv3)
        skeletonSideOut4 = self.skeletonSideOut4(conv4)
        skeletonSideOut5 = self.skeletonSideOut5(conv5)

        skeletonUpsample2 = torch.nn.functional.conv_transpose2d(skeletonSideOut2, self.skeletonWeightDeconv2, stride=2)
        skeletonUpsample3 = torch.nn.functional.conv_transpose2d(skeletonSideOut3, self.skeletonWeightDeconv3, stride=4)
        skeletonUpsample4 = torch.nn.functional.conv_transpose2d(skeletonSideOut4, self.skeletonWeightDeconv4, stride=8)
        skeletonUpsample5 = torch.nn.functional.conv_transpose2d(skeletonSideOut5, self.skeletonWeightDeconv5, stride=16)

        # Aligned cropping.
        skeletonSideOut2 = skeletonUpsample2[:, :, self.crop2_margin:self.crop2_margin+height, self.crop2_margin:self.crop2_margin+width]
        skeletonSideOut3 = skeletonUpsample3[:, :, self.crop3_margin:self.crop3_margin+height, self.crop3_margin:self.crop3_margin+width]
        skeletonSideOut4 = skeletonUpsample4[:, :, self.crop4_margin:self.crop4_margin+height, self.crop4_margin:self.crop4_margin+width]
        skeletonSideOut5 = skeletonUpsample5[:, :, self.crop5_margin:self.crop5_margin+height, self.crop5_margin:self.crop5_margin+width]

        skeletonFuseScale0 = torch.cat((skeletonSideOut2[:,0:1,:,:], skeletonSideOut3[:,0:1,:,:], skeletonSideOut4[:,0:1,:,:], skeletonSideOut5[:,0:1,:,:] ),1)
        skeletonFuseScale1 = torch.cat((skeletonSideOut2[:,1:2,:,:], skeletonSideOut3[:,1:2,:,:], skeletonSideOut4[:,1:2,:,:], skeletonSideOut5[:,1:2,:,:] ),1)
        skeletonFuseScale2 = torch.cat((skeletonSideOut3[:,2:3,:,:], skeletonSideOut4[:,2:3,:,:], skeletonSideOut5[:,2:3,:,:] ),1)
        skeletonFuseScale3 = torch.cat((skeletonSideOut4[:,3:4,:,:], skeletonSideOut5[:,3:4,:,:] ),1)
        skeletonFuseScale4 = skeletonSideOut5[:,4:5,:,:]

        skeletonFuseScale0 = self.skeletonFuseScale0(skeletonFuseScale0)
        skeletonFuseScale1 = self.skeletonFuseScale1(skeletonFuseScale1)
        skeletonFuseScale2 = self.skeletonFuseScale2(skeletonFuseScale2)
        skeletonFuseScale3 = self.skeletonFuseScale3(skeletonFuseScale3)
        
        skeletonFuse = torch.cat((skeletonFuseScale0,skeletonFuseScale1, skeletonFuseScale2, skeletonFuseScale3, skeletonFuseScale4), 1)
        

        return edgeSideOut1, edgeSideOut2, edgeSideOut3, edgeSideOut4, edgeSideOut5, edgeFuse, skeletonSideOut2, skeletonSideOut3, skeletonSideOut4, skeletonSideOut5, skeletonFuse

def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w