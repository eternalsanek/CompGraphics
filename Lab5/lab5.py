import numpy as np
from PIL import Image

def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangle_normal(P0, P1, P2):
    v1 = P1 - P2
    v2 = P1 - P0

    N = np.cross(v1, v2)

    if np.linalg.norm(N) > 0:
        N = N / np.linalg.norm(N)

    return N

def draw_triangle_texture_and_shading(k, height, width, x0, y0, z0, x1, y1, z1, x2, y2, z2, v_t0, v_t1, v_t2, n0, n1, n2,
                          matrix, z_buffer, texture, H_T, W_T):
    x0_s = k * x0 / z0 + width / 2
    y0_s = k * y0 / z0 + height / 2
    x1_s = k * x1 / z1 + width / 2
    y1_s = k * y1 / z1 + height / 2
    x2_s = k * x2 / z2 + width / 2
    y2_s = k * y2 / z2 + height / 2

    xmin = int(min(x0_s, x1_s, x2_s))
    if xmin < 0:
        xmin = 0
    xmax = int(max(x0_s, x1_s, x2_s)) + 1
    if xmax > width:
        xmax = width
    ymin = int(min(y0_s, y1_s, y2_s))
    if ymin < 0:
        ymin = 0
    ymax = int(max(y0_s, y1_s, y2_s)) + 1
    if ymax > height:
        ymax = height

    I0 = svet(n0)
    I1 = svet(n1)
    I2 = svet(n2)

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(i, j, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                I = -(I0 * lambda0 + I1 * lambda1 + I2 * lambda2)
                if z < z_buffer[j, i] and I > 0:
                    # I = - (I0 * lambda0 + I1 * lambda1 + I2 * lambda2)
                    v = round(H_T * (1 - (lambda0 * v_t0[1] + lambda1 * v_t1[1] + lambda2 * v_t2[1])))
                    u = round(W_T * (lambda0 * v_t0[0] + lambda1 * v_t1[0] + lambda2 * v_t2[0]))
                    # print(v, u)
                    rgb = texture[v, u] * I
                    z_buffer[j, i] = z
                    matrix[j, i] = rgb

def draw_triangle_shading(k, height, width, x0, y0, z0, x1, y1, z1, x2, y2, z2, n0, n1, n2, matrix, z_buffer):
    x0_s = k * x0 / z0 + width / 2
    y0_s = k * y0 / z0 + height / 2
    x1_s = k * x1 / z1 + width / 2
    y1_s = k * y1 / z1 + height / 2
    x2_s = k * x2 / z2 + width / 2
    y2_s = k * y2 / z2 + height / 2

    xmin = int(min(x0_s, x1_s, x2_s))
    if xmin < 0:
        xmin = 0
    xmax = int(max(x0_s, x1_s, x2_s)) + 1
    if xmax > width:
        xmax = width
    ymin = int(min(y0_s, y1_s, y2_s))
    if ymin < 0:
        ymin = 0
    ymax = int(max(y0_s, y1_s, y2_s)) + 1
    if ymax > height:
        ymax = height

    I0 = svet(n0)
    I1 = svet(n1)
    I2 = svet(n2)

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(i, j, x0_s, y0_s, x1_s, y1_s, x2_s, y2_s)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                I = -(I0 * lambda0 + I1 * lambda1 + I2 * lambda2)
                if z < z_buffer[j, i] and I > 0:
                    # print(v, u)
                    rgb = 255 * I
                    z_buffer[j, i] = z
                    matrix[j, i] = rgb

def svet(n):
    l = [0, 0, 1]
    n = n / np.linalg.norm(n)
    return np.dot(n, l)

def quat_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def model_rotation_quat(v, v_n, alpha, betta, gamma, t):
    ca, sa = np.cos(-alpha / 2), np.sin(-alpha / 2)
    cb, sb = np.cos(betta / 2), np.sin(betta / 2)
    cg, sg = np.cos(-gamma / 2), np.sin(-gamma / 2)

    qx = np.array([ca, sa, 0, 0])
    qy = np.array([cb, 0, sb, 0])
    qz = np.array([cg, 0, 0, sg])

    q = quat_mult(quat_mult(qx, qy), qz)
    q = q / np.linalg.norm(q)

    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z])

    for i in range(len(v)):
        a, b, c = v[i]
        p = np.array([0, a, b, c])
        p_rot = quat_mult(quat_mult(q, p), q_conj)
        v[i] = p_rot[1:] + t

    for i in range(len(v_n)):
        a, b, c = v_n[i]
        p = np.array([0, a, b, c])
        p_rot = quat_mult(quat_mult(q, p), q_conj)
        v_n[i] = p_rot[1:]

def model_rotation_quat_matrix(v, v_n, alpha, betta, gamma, t):
    ca, sa = np.cos(-alpha / 2), np.sin(-alpha / 2)
    cb, sb = np.cos(betta / 2), np.sin(betta / 2)
    cg, sg = np.cos(-gamma / 2), np.sin(-gamma / 2)

    qx = np.array([ca, sa, 0, 0])
    qy = np.array([cb, 0, sb, 0])
    qz = np.array([cg, 0, 0, sg])

    q = quat_mult(quat_mult(qx, qy), qz)
    q = q / np.linalg.norm(q)
    a, b, c, d = q

    R = np.array([[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c ],
                  [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b],
                  [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2]])

    print("Матрица поворота из model_roration_quat_matrix:\n", R)

    for i in range(len(v)):
        v[i] = np.dot(R, v[i]) + t

    for i in range(len(v_n)):
        v_n[i] = np.dot(R, v_n[i])

def model_rotation_euler(v, v_n, alpha, betta, gamma, t):
    matrix1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
    matrix2 = np.array([[np.cos(betta), 0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])
    matrix3 = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R = np.dot(np.dot(matrix1, matrix2), matrix3)

    print("Матрица поворота из model_rotation_euler:\n", R)

    for i in range(len(v)):
        v[i] = np.dot(R, v[i]) + t

    for i in range(len(v_n)):
        v_n[i] = np.dot(R, v_n[i])

def parser_obj(filename, v, v_t, v_n, f_v, f_vt, f_vn):
    file = open(filename)
    for s in file:
        sp = s.split()
        if not sp:
            continue
        if sp[0] == 'v':
            v.append([float(sp[1]), float(sp[2]), float(sp[3])])
        elif sp[0] == 'vt':
            v_t.append([float(sp[1]), float(sp[2])])
        elif sp[0] == 'vn':
            # print('типо vn не запоминаем')
            v_n.append([float(sp[1]), float(sp[2]), float(sp[3])])
        elif sp[0] == 'f':
            l = len(sp)
            for i in range(2, l - 1):
                f_v.append([int(sp[1].split('/')[0]) - 1,
                    int(sp[i].split('/')[0]) - 1,
                    int(sp[i + 1].split('/')[0]) - 1])
                if len(sp[1].split('/')) == 2:
                    f_vt.append([int(sp[1].split('/')[1]) - 1,
                                 int(sp[i].split('/')[1]) - 1,
                                 int(sp[i + 1].split('/')[1]) - 1])
                if len(sp[1].split('/')) == 3 and sp[1].split('/')[1] == '':
                    f_vn.append([int(sp[1].split('/')[2]) - 1,
                                 int(sp[i].split('/')[2]) - 1,
                                 int(sp[i + 1].split('/')[2]) - 1])
                elif   len(sp[1].split('/')) == 3:
                    f_vt.append([int(sp[1].split('/')[1]) - 1,
                                 int(sp[i].split('/')[1]) - 1,
                                 int(sp[i + 1].split('/')[1]) - 1])
                    f_vn.append([int(sp[1].split('/')[2]) - 1,
                                 int(sp[i].split('/')[2]) - 1,
                                 int(sp[i + 1].split('/')[2]) - 1])
    file.close()

def rendering(v, v_t, v_n, f_v, f_vt, f_vn, matrix, z_buffer, height_image, width_image, koeff, alpha,
              betta, gamma, t, texture, H_T, W_T):

    exist_vn = True
    if len(v_n) == 0:
        exist_vn = False

        for s in range (len(v)):
            v_n.append([0, 0, 0])

        for k in range(len(f_v)):
            i0, i1, i2 = f_v[k]
            x0, y0, z0 = v[i0]
            x1, y1, z1 = v[i1]
            x2, y2, z2 = v[i2]

            P0 = np.array([x0, y0, z0])
            P1 = np.array([x1, y1, z1])
            P2 = np.array([x2, y2, z2])

            N = triangle_normal(P0, P1, P2)
            v_n[i0] += N
            v_n[i1] += N
            v_n[i2] += N

        for k in range(len(v_n)):
            if np.linalg.norm(v_n[k]) > 0:
                v_n[k] /= np.linalg.norm(v_n[k])

    model_rotation_quat_matrix(v, v_n, alpha, betta, gamma, t)

    if texture is not None:
        for k in range(len(f_v)):
            i0, i1, i2 = f_v[k]
            x0, y0, z0 = v[i0]
            x1, y1, z1 = v[i1]
            x2, y2, z2 = v[i2]

            # для тонировки Гуро
            if exist_vn:
                n0 = v_n[f_vn[k][0]]
                n1 = v_n[f_vn[k][1]]
                n2 = v_n[f_vn[k][2]]
            else:
                n0 = v_n[i0]
                n1 = v_n[i1]
                n2 = v_n[i2]

            v_t0 = v_t[f_vt[k][0]]
            v_t1 = v_t[f_vt[k][1]]
            v_t2 = v_t[f_vt[k][2]]

            draw_triangle_texture_and_shading(koeff,  height_image, width_image,  x0, y0, z0, x1, y1, z1, x2, y2, z2,
                          v_t0, v_t1, v_t2, n0, n1, n2, matrix, z_buffer,texture, H_T, W_T)
            print(k/len(f_v) * 100, '%')
    else:
        for k in range(len(f_v)):
            i0, i1, i2 = f_v[k]
            x0, y0, z0 = v[i0]
            x1, y1, z1 = v[i1]
            x2, y2, z2 = v[i2]

            # для тонировки Гуро
            if exist_vn:
                n0 = v_n[f_vn[k][0]]
                n1 = v_n[f_vn[k][1]]
                n2 = v_n[f_vn[k][2]]
            else:
                n0 = v_n[i0]
                n1 = v_n[i1]
                n2 = v_n[i2]

            draw_triangle_shading(koeff,  height_image, width_image, x0, y0, z0, x1, y1, z1, x2, y2, z2,
                                  n0, n1, n2, matrix, z_buffer)
            print(k / len(f_v) * 100, '%')
    print('rendering done')

height, width = 1000, 300
img_mat = np.zeros((height, width, 3), dtype=np.uint8)
z_buffer = np.full((height, width), np.inf)

v0 = []
v_t0 = []
v_n0 = []
f_v0 = []
f_vt0 = []
f_vn0 = []

parser_obj('12268_banjofrog_v1_L3.obj', v0, v_t0, v_n0, f_v0, f_vt0, f_vn0)
texture0 = None
H_T0 = None
W_T0 = None

rendering(v0, v_t0, v_n0, f_v0, f_vt0, f_vn0, img_mat, z_buffer, height, width, 1000, 4.8 * np.pi / 4, 0,
           0, [0, 0, 6.5], texture0, H_T0, W_T0)

v = []
v_t = []
v_n = []
f_v = []
f_vt = []
f_vn = []

parser_obj('model.obj', v, v_t, v_n, f_v, f_vt, f_vn)

texture_img = Image.open('bunny-atlas.jpg').convert('RGB')
texture = np.array(texture_img)
H_T, W_T = texture.shape[:2]

rendering(v, v_t, v_n, f_v, f_vt, f_vn, img_mat, z_buffer, height, width, 12000, np.pi, np.pi / 2,
          0, [0.04, 0.07, 1], texture, H_T, W_T)

v1 = []
v_t1 = []
v_n1 = []
f_v1 = []
f_vt1 = []
f_vn1 = []

parser_obj('12268_banjofrog_v1_L3.obj', v1, v_t1, v_n1, f_v1, f_vt1, f_vn1)

texture_img1 = Image.open('12268_banjofrog_diffuse.jpg').convert('RGB')
texture1 = np.array(texture_img1)
H_T1, W_T1 = texture1.shape[:2]

rendering(v1, v_t1, v_n1, f_v1, f_vt1, f_vn1, img_mat, z_buffer, height, width, 1000, 4.8 * np.pi / 4, 0,
          0, [-0.9, 0.7, 6.5], texture1, H_T1, W_T1)

v2 = []
v_t2 = []
v_n2 = []
f_v2 = []
f_vt2 = []
f_vn2 = []

parser_obj('12221_Cat_v1_l3.obj', v2, v_t2, v_n2, f_v2, f_vt2, f_vn2)

texture_img2 = Image.open('Cat_diffuse.jpg').convert('RGB')
texture2 = np.array(texture_img2)
H_T2, W_T2 = texture2.shape[:2]

rendering(v2, v_t2, v_n2, f_v2, f_vt2, f_vn2, img_mat, z_buffer, height, width, 2000, 4.48 * np.pi / 4, 0,
          -np.pi / 4, [-7, -6, 50], texture2, H_T2, W_T2)

image = Image.fromarray(img_mat, mode="RGB")
image.save("test18_quaterion_matrix.png")