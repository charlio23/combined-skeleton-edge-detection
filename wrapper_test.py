from pix2pix import Wrapper
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

w = Wrapper()

transf = transforms.ToTensor()

edge_img = 2*transf(Image.open("test_skeleton.png").convert('L')).unsqueeze_(0) - 1
ske_img = 2*transf(Image.open("test_edge.png").convert('L')).unsqueeze_(0) - 1
ske_img = torch.cat([ske_img, (ske_img > -1).float()*2 - 1.0], 1)
edge_input = {'A': edge_img, 'B': ske_img, 'A_paths': None}
ske_input = {'A': ske_img, 'B': edge_img, 'A_paths': None}

skeleton = (w.edge_to_skeleton(edge_input) + 1)/2
edge = (w.skeleton_to_edge(ske_input) + 1)/2

print(torch.max(skeleton), torch.min(skeleton))

plt.imshow(grayTrans(skeleton))
plt.show()

plt.imshow(grayTrans(edge))
plt.show()