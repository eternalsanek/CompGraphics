import numpy as np
from PIL import Image

from Task2 import *

file = open('model.obj')

v = []
f = []

for s in file:
    sp = s.split()
    if sp[0] == 'v':
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])
    elif sp[0] == 'f':
        f.append([int(sp[1].split('/')[0]) - 1,
                  int(sp[2].split('/')[0]) - 1,
                  int(sp[3].split('/')[0]) - 1])

file.close()

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)

for k in range(len(f)):
    x0 = 10000 * v[f[k][0]][0] + 750
    y0 = 10000 * v[f[k][0]][1] + 750
    x1 = 10000 * v[f[k][1]][0] + 750
    y1 = 10000 * v[f[k][1]][1] + 750
    x2 = 10000 * v[f[k][2]][0] + 750
    y2 = 10000 * v[f[k][2]][1] + 750

    bresenham_line(img_mat, int(x0), int(y0), int(x1), int(y1), 255)
    bresenham_line(img_mat, int(x1), int(y1), int(x2), int(y2), 255)
    bresenham_line(img_mat, int(x2), int(y2), int(x0), int(y0), 255)

image = Image.fromarray(img_mat, mode = "RGB")
image.save("rabbit.png")