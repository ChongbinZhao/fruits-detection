from PIL import Image
import cv2
import numpy as np
import pywt


def NRG(x, y, image):  # image是图像读取操作后得到的矩阵,cv2按照BGR的顺序读取图片
    R = image[x][y][2]
    G = image[x][y][1]
    Y = (R+G)
    nrg = (Y - 2*G) / (Y + 2*G)
    return 255 * (nrg + 1) / 2


def tomato_location():
    SRC = cv2.imread('orange.png')
    src = SRC.astype(np.float32)
    # cv2.imshow('Source', SRC)
    # cv2.waitKey(0)

    [N, M] = SRC.shape[:2]  # 获取图片长宽
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # gray = gray.astype(np.uint8)
    # cv2.imwrite('gray_0.png', gray)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    # 对灰度图进行红绿色差法归一化，主要是显著突出原图红色与绿色的差异，然后以灰度图的形式呈现
    for x in range(N):
        for y in range(M):
            gray[x][y] = NRG(x, y, src)

    gray = gray.astype(np.uint8)
    cv2.imwrite('binary2.png', gray)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    _, img1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('binary.png', img1)
    # cv2.imshow('OTSU', img1)
    # cv2.waitKey(0)

    # 第一次孔洞填充
    im_floodfill = img1.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img2 = img1 | im_floodfill_inv
    # cv2.imshow('Image filling', img2)
    # cv2.waitKey(0)

    # 开运算
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * 2 - 1, 6 * 2 - 1))  # 椭圆状的11x11掩膜
    img3 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel_open)
    # cv2.imshow('opening circle', img3)
    # cv2.waitKey(0)

    # 闭运算
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * 2 - 1, 8 * 2 - 1))  # 椭圆状的15x15掩膜
    img4 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel_close)
    # 第二次孔洞填充
    im_floodfill = img4.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img4 = img4 | im_floodfill_inv
    # cv2.imshow('closing circle', img4)
    # cv2.waitKey(0)

    # 在原图中提取番茄所在区域
    img5 = src
    for x in range(N):
        for y in range(M):
            if img4[x][y] == 0:
                img5[x][y][:] = 0
    img5 = img5.astype(np.uint8)
    # cv2.imshow('tomato area', img5)
    # cv2.waitKey(0)

    # 提前到的番茄区域转为灰度图
    img6 = cv2.cvtColor(img5, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray tomato area', img6)
    # cv2.waitKey(0)

    # 小波变换图像重构，增强图像边缘
    cA, (cH, cV, cD) = pywt.dwt2(img6, "haar")
    img7 = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    # cv2.imshow('haar', img7)
    # cv2.waitKey(0)

    img7 = img7.astype(np.uint8)
    [N, M] = img7.shape[:2]
    # 第三次孔洞填充
    im_floodfill = img7.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img7 = img7 | im_floodfill_inv

    # canny算子提取边缘
    blurred = cv2.GaussianBlur(img7, (11, 11), 0)
    img8 = cv2.Canny(blurred, 10, 70)
    # cv2.imshow('canny', img8)
    # cv2.waitKey(0)
    img8 = Image.fromarray(img8)
    img8.save('canny.png')

    img9 = cv2.imread('canny.png')
    img10 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
    # img10 = img10.astype(np.uint8)
    # img10 = img10.astype(np.float32)

    # 霍夫圆检测，检测番茄的位置（番茄在平面的形状比较接近一个圆）
    circles0 = cv2.HoughCircles(img10, cv2.HOUGH_GRADIENT, 1,
                                40, param1=255, param2=9, minRadius=80, maxRadius=105)
    circles = circles0[0, :, :]
    circles = np.uint16(np.around(circles))
    for i in circles[:]:
        cv2.circle(SRC, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
        cv2.circle(SRC, (i[0], i[1]), 2, (0, 0, 0), 3)  # 画圆心

    # img10 = img10.astype(np.float32)
    cv2.imwrite('result.png', SRC)
    cv2.imshow('result', SRC)
    cv2.waitKey(0)


if __name__ == '__main__':
    tomato_location()
