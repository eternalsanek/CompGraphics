import numpy as np
from PIL import Image

img_mat = np.zeros((200, 200, 3), dtype=np.uint8)

def draw_line1(img_matrix, x0, y0, x1, y1, color):
    count = 100
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img_matrix[y, x] = color

def draw_line2(img_matrix, x0, y0, x1, y1, color):
    count = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img_matrix[y, x] = color

def draw_line3(img_matrix, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            img_matrix[x, y] = color
        else:
            img_matrix[y, x] = color

def draw_line4(img_matrix, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if (xchange):
            img_matrix[x, y] = color
        else:
            img_matrix[y, x] = color

        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update

def bresenham_line(img_matrix, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if (xchange):
            img_matrix[x, y] = color
        else:
            img_matrix[y, x] = color

        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

for k in range (13):
    x0, y0 = 100, 100
    x1 = int(100 + 95 * np.cos(2 * np.pi / 13 * k))
    y1 = int(100 + 95 * np.sin(2 * np.pi / 13 * k))
    bresenham_line(img_mat, x0, y0, x1, y1, 255)

image = Image.fromarray(img_mat, mode = "RGB")
image.save("bresenham.png")