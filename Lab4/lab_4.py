import numpy as np
from PIL import Image


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def draw_triangle_shading(k, b, x0, y0, z0, x1, y1, z1, x2, y2, z2, n0, n1, n2, matrix, z_buffer):
    x0_s = 10000 * x0 / z0 + 1000
    y0_s = 10000 * y0 / z0 + 1000
    x1_s = 10000 * x1 / z1 + 1000
    y1_s = 10000 * y1 / z1 + 1000
    x2_s = 10000 * x2 / z2 + 1000
    y2_s = 10000 * y2 / z2 + 1000

    xmin = int(min(x0_s, x1_s, x2_s))
    if xmin < 0:
        xmin = 0
    xmax = int(max(x0_s, x1_s, x2_s)) + 1
    if xmax > 1999:
        xmax = 1999 + 1
    ymin = int(min(y0_s, y1_s, y2_s))
    if ymin < 0:
        ymin = 0
    ymax = int(max(y0_s, y1_s, y2_s)) + 1
    if ymax > 1999:
        ymax = 1999 + 1

    I0 = svet(n0)
    I1 = svet(n1)
    I2 = svet(n2)

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(i, j, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buffer[j, i]:
                    I = - (I0 * lambda0 + I1 * lambda1 + I2 * lambda2)
                    color = [255, 0, 0]
                    z_buffer[j, i] = z
                    matrix[j, i] = color * I

def draw_triangle_texture(k, b, x0, y0, z0, x1, y1, z1, x2, y2, z2, v_t0, v_t1, v_t2, n0, n1, n2, matrix, z_buffer, texture, H_T, W_T):
    x0_s = k * x0 / z0 + b
    y0_s = k * y0 / z0 + b
    x1_s = k * x1 / z1 + b
    y1_s = k * y1 / z1 + b
    x2_s = k * x2 / z2 + b
    y2_s = k * y2 / z2 + b

    xmin = int(min(x0_s, x1_s, x2_s))
    if xmin < 0:
        xmin = 0
    xmax = int(max(x0_s, x1_s, x2_s)) + 1
    if xmax > 1999:
        xmax = 1999 + 1
    ymin = int(min(y0_s, y1_s, y2_s))
    if ymin < 0:
        ymin = 0
    ymax = int(max(y0_s, y1_s, y2_s)) + 1
    if ymax > 1999:
        ymax = 1999 + 1

    I0 = svet(n0)
    I1 = svet(n1)
    I2 = svet(n2)

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(i, j, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buffer[j, i]:
                    I = - (I0 * lambda0 + I1 * lambda1 + I2 * lambda2)
                    v = round(H_T * (1 - (lambda0 * v_t0[1] + lambda1 * v_t1[1] + lambda2 * v_t2[1])))
                    u = round(W_T * (lambda0 * v_t0[0] + lambda1 * v_t1[0] + lambda2 * v_t2[0]))
                    # print(v, u)
                    rgb = texture[v, u] * I
                    z_buffer[j, i] = z
                    matrix[j, i] = rgb

def triangle_normal(P0, P1, P2):
    v1 = P1 - P2
    v2 = P1 - P0

    N = np.cross(v1, v2)

    if np.linalg.norm(N) > 0:
        N = N / np.linalg.norm(N)

    return N

def cos_of_the_angle(N):
    l = np.array([0, 0, 1])
    norm = np.linalg.norm(N)
    return np.dot(N, l) / norm

# вектор n уже нормирован
def svet(n):
    l = [0, 0, 1]
    return np.dot(n, l)

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
z_buffer = np.full((2000, 2000), np.inf)

texture_img = Image.open('bunny-atlas.jpg').convert('RGB')
texture = np.array(texture_img)
H_T, W_T = texture.shape[:2]

filename = '../Lab3/model.obj'
file = open(filename)
v = []
f_v = []
f_vt = []
v_n = []
v_t = []

for s in file:
    sp = s.split()
    if sp[0] == 'v':
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])
        v_n.append([0, 0, 0])
    elif sp[0] == 'f':
        f_v.append([int(sp[1].split('/')[0]) - 1,
                    int(sp[2].split('/')[0]) - 1,
                    int(sp[3].split('/')[0]) - 1])
        f_vt.append([int(sp[1].split('/')[1]) - 1,
                     int(sp[2].split('/')[1]) - 1,
                     int(sp[3].split('/')[1]) - 1])
    elif sp[0] == 'vt':
        v_t.append([float(sp[1]), float(sp[2])])

alpha = np.pi
betta = np.pi / 2
gamma = 0
t = [-0.004, -0.0024, 1]
matrix1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
matrix2 = np.array([[np.cos(betta), 0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])
matrix3 = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
R = np.dot(np.dot(matrix1, matrix2), matrix3)

for i in range(len(v)):
    v[i] = np.dot(R, v[i]) + t

for k in range(len(f_v)):
    x0 = v[f_v[k][0]][0]
    y0 = v[f_v[k][0]][1]
    z0 = v[f_v[k][0]][2]
    x1 = v[f_v[k][1]][0]
    y1 = v[f_v[k][1]][1]
    z1 = v[f_v[k][1]][2]
    x2 = v[f_v[k][2]][0]
    y2 = v[f_v[k][2]][1]
    z2 = v[f_v[k][2]][2]

    P0 = np.array([x0, y0, z0])
    P1 = np.array([x1, y1, z1])
    P2 = np.array([x2, y2, z2])

    N = triangle_normal(P0, P1, P2)
    v_n[f_v[k][0]] += N
    v_n[f_v[k][1]] += N
    v_n[f_v[k][2]] += N

for s in range(len(v_n)):
    if np.linalg.norm(v_n[s]) > 0:
        v_n[s] /= np.linalg.norm(v_n[s])

for k in range(len(f_v)):
    x0 = v[f_v[k][0]][0]
    y0 = v[f_v[k][0]][1]
    z0 = v[f_v[k][0]][2]
    x1 = v[f_v[k][1]][0]
    y1 = v[f_v[k][1]][1]
    z1 = v[f_v[k][1]][2]
    x2 = v[f_v[k][2]][0]
    y2 = v[f_v[k][2]][1]
    z2 = v[f_v[k][2]][2]

    v_t0 = v_t[f_vt[k][0]]
    v_t1 = v_t[f_vt[k][1]]
    v_t2 = v_t[f_vt[k][2]]

    # для тонировки Гуро
    n0 = v_n[f_v[k][0]]
    n1 = v_n[f_v[k][1]]
    n2 = v_n[f_v[k][2]]
    # draw_triangle_shading(10000, 1000, x0, y0, z0, x1, y1, z1, x2, y2, z2, n0, n1, n2, img_mat, z_buffer)
    draw_triangle_texture(10000, 1000, x0, y0, z0, x1, y1, z1, x2, y2, z2, v_t0, v_t1, v_t2, n0, n1, n2,
                          img_mat, z_buffer, texture, H_T, W_T)

image = Image.fromarray(img_mat, mode="RGB")
image.save("task18_texture.png")