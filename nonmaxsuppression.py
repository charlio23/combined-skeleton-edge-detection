from scipy import ndimage
import numpy as np
import torch

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Kx = torch.from_numpy(np.ascontiguousarray(Kx[::-1, ::-1])).unsqueeze_(0).unsqueeze_(0)
    Ky = torch.from_numpy(np.ascontiguousarray(Ky[::-1, ::-1])).unsqueeze_(0).unsqueeze_(0)

    Ix = torch.nn.functional.conv2d(img, Kx, padding=1)
    Iy = torch.nn.functional.conv2d(img, Ky, padding=1)
    
    #G = np.hypot(Ix, Iy)
    #G = G / G.max() * 255
    #theta = np.arctan2(Iy, Ix)
    
    return Ix, Iy

def nonmax(img, O, r, m):
    h, w = img.shape
    edge = img.clone()
    e = img.clone()*m
    iMat = torch.t(torch.arange(h).float()*torch.ones((w,1)))
    jMat = torch.arange(w).float()*torch.ones((h,1))
    index = torch.cat([iMat.unsqueeze(-1), jMat.unsqueeze(-1)], -1)
    cosO, sinO = torch.cos(O.float()), torch.sin(O.float())
    angle = torch.cat([cosO.unsqueeze(-1), sinO.unsqueeze(-1)], -1)
    for d in r:
        ort = index + d*angle
        xy = torch.cat([ort[:,:,:1].clamp(0,h - 1.001),ort[:,:,1:].clamp(0,w - 1.001)], -1)
        xy0 = xy.long()
        xy1 = xy0.add(1)
        dxy0 = xy - xy0.float()
        dxy1 = 1 - dxy0
        x1y0 = torch.cat([xy1[:,:,:1], xy0[:,:,1:]], -1)
        x0y1 = torch.cat([xy0[:,:,:1], xy1[:,:,1:]], -1)
        res1 = img[xy0.reshape(-1,2).t().chunk(chunks=2,dim=0)].reshape(h,w)*dxy1[:,:,0]*dxy1[:,:,1]
        res2 = img[x1y0.reshape(-1,2).t().chunk(chunks=2,dim=0)].reshape(h,w)*dxy0[:,:,0]*dxy1[:,:,1]
        res3 = img[x0y1.reshape(-1,2).t().chunk(chunks=2,dim=0)].reshape(h,w)*dxy1[:,:,0]*dxy0[:,:,1]
        res4 = img[xy1.reshape(-1,2).t().chunk(chunks=2,dim=0)].reshape(h,w)*dxy0[:,:,0]*dxy0[:,:,1]
        e0 = res1 + res2 + res3 + res4
        edge[(e<e0)] = 0
    return edge

def nms(img):
    img = img.unsqueeze_(0).float()
    Ox, Oy = sobel_filters(img)
    Oxx, _ = sobel_filters(Ox)
    Oxy, Oyy = sobel_filters(Oy)
    O = torch.remainder(torch.atan(torch.div(torch.mul(Oyy, torch.sign(-Oxy)),(Oxx + 1e-5))), np.pi).squeeze_(0).squeeze_(0).to(torch.uint8)
    img = img.squeeze_(0).squeeze_(0)
    E = nonmax(img, O, [-1, 1], 1.01)

    return E.unsqueeze_(0).unsqueeze_(0)