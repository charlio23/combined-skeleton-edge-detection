from scipy import ndimage
import numpy as np

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    #G = np.hypot(Ix, Iy)
    #G = G / G.max() * 255
    #theta = np.arctan2(Iy, Ix)
    
    return Ix, Iy

def interp(img, h, w, x , y):
    if x < 0:
        x = 0
    elif x > h - 1.001:
        x = h - 1.001
    if y < 0:
        y = 0
    elif y > w - 1.001:
        y = w - 1.001
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    dx0, dy0 = float(x) - float(x0), float(y) - float(y0)
    dx1, dy1 = 1 - dx0, 1 - dy0
    return img[x0,y0]*dx1*dy1 + img[x1,y0]*dx0*dy1 + img[x0,y1]*dx1*dy0 + img[x1,y1]*dx0*dy0;


def nonmax(img, O, r, s, m):
    h, w = img.shape
    edge = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            e = edge[i,j] = img[i,j]
            if (e < 1e-4):
                continue
            e*= m
            coso, sino = np.cos(O[i,j]), np.sin(O[i,j])
            for d in range(-r, r+1):
                if d:
                    e0 = interp(img, h, w, i+d*coso, j+d*sino)
                    if e < e0:
                        edge[i,j] = 0
                        break
    return edge



def nms(img):
    Ox, Oy = sobel_filters(img)
    Oxx, _ = sobel_filters(Ox)
    Oxy, Oyy = sobel_filters(Oy)
    O = np.remainder(np.arctan(np.divide(np.multiply(Oyy, np.sign(-Oxy)),(Oxx + 1e-5))), np.pi)
    E = nonmax(img, O, 1, 5, 1.01)
    return E