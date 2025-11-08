import numpy as np
from PIL import Image


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, matrix, z_buffer, rgb):
    x0_s = 1000 * x0 / z0 + 1000
    y0_s = 1000 * y0 / z0 + 1000
    x1_s = 1000 * x1 / z1 + 1000
    y1_s = 1000 * y1 / z1 + 1000
    x2_s = 1000 * x2 / z2 + 1000
    y2_s = 1000 * y2 / z2 + 1000

    xmin = int(min(x0_s, x1_s, x2_s))
    if xmin < 0:
        xmin = 0
    xmax = int(max(x0_s, x1_s, x2_s)) + 1
    if xmax > 1999:
        xmax = 1999
    ymin = int(min(y0_s, y1_s, y2_s))
    if ymin < 0:
        ymin = 0
    ymax = int(max(y0_s, y1_s, y2_s)) + 1
    if ymax > 1999:
        ymax = 1999

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(i, j, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buffer[j, i]:
                    z_buffer[j, i] = z
                    matrix[j, i] = rgb


def triangle_normal(P0, P1, P2):
    v1 = P1 - P2
    v2 = P1 - P0

    N = np.cross(v1, v2)
    return N


def angle_of_incidence(N):
    l = np.array([0, 0, 1])
    norm = np.linalg.norm(N)

    return np.dot(N, l) / norm


img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
z_buffer = np.full((2000, 2000), np.inf)

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

alpha = np.pi / 2
betta = np.pi / 4
gamma = 0
t = [-0.004, -0.0024, 0.1]
matrix1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
matrix2 = np.array([[np.cos(betta), 0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])
matrix3 = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
R = np.dot(np.dot(matrix1, matrix2), matrix3)

for i in range(len(v)):
    v[i] = np.dot(R, v[i]) + t

for k in range(len(f)):
    x0 = v[f[k][0]][0]
    y0 = v[f[k][0]][1]
    z0 = v[f[k][0]][2]
    x1 = v[f[k][1]][0]
    y1 = v[f[k][1]][1]
    z1 = v[f[k][1]][2]
    x2 = v[f[k][2]][0]
    y2 = v[f[k][2]][1]
    z2 = v[f[k][2]][2]

    P0 = np.array([x0, y0, z0])
    P1 = np.array([x1, y1, z1])
    P2 = np.array([x2, y2, z2])
    N = triangle_normal(P0, P1, P2)
    m = angle_of_incidence(N)
    if (m < 0):
        draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat, z_buffer, [-255 * m, 0, 0])

image = Image.fromarray(img_mat, mode="RGB")
image.save("task15_0.1a.png")