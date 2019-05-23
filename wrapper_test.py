from pix2pix import Wrapper
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def grayTrans(img):
    img = img.data.cpu().numpy()[0][0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

w = Wrapper()

transf = transforms.ToTensor()

edge_img = transf(Image.open("test_skeleton.png").convert('L')).unsqueeze_(0)
ske_img = transf(Image.open("test_edge.png").convert('L')).unsqueeze_(0)

edge_input = {'A': edge_img, 'B': ske_img, 'A_paths': None}
ske_input = {'A': ske_img, 'B': edge_img, 'A_paths': None}

skeleton = w.edge_to_skeleton(edge_input)
edge = w.skeleton_to_edge(ske_input)

plt.imshow(grayTrans(skeleton))
plt.show()

plt.imshow(grayTrans(edge))
plt.show()