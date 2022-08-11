# -*- coding: utf-8 -*-
# @Time : $2022.1.7
# @Author : 赵崇斌
# @Class  : 智能科学与技术2班
# @File : fruit_location.py

from PIL import Image
import cv2
import numpy as np
import pywt
import math


def NRG(x, y, image):  # 红色分量和绿色分量对比度
    R = image[x][y][2]
    G = image[x][y][1]
    nrg = (R - G) / (R + G)
    return 255 * (nrg + 1) / 2


def NYG(x, y, image):  # 黄色分量和绿色分量对比度
    R = image[x][y][2]
    G = image[x][y][1]
    Y = (R + G) / 2
    nrg = (Y - G) / (Y + G)
    return 255 * (nrg + 1) / 2


def tomato_location(path):
    SRC = cv2.imread(path)  # 读取图片
    src = SRC.astype(np.float32)    # 图片原来是uint8类型的，用于处理的话要转为float类型
    # cv2.imshow('Source', SRC)
    # cv2.waitKey(0)

    [N, M] = SRC.shape[:2]  # 获取图片长宽
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    # 灰度化
    # gray = gray.astype(np.uint8)
    cv2.imwrite('gray_0.png', gray)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    # 对灰度图进行红绿色差法归一化，主要是显著突出原图红色与绿色的差异，然后以灰度图的形式呈现
    for x in range(N):
        for y in range(M):
            gray[x][y] = NRG(x, y, src)

    gray = gray.astype(np.uint8)
    cv2.imwrite('gray_nrg.png', gray)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    _, img1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # Otsu法二值化
    cv2.imwrite('binary.png', img1)
    # cv2.imshow('OTSU', img1)
    # cv2.waitKey(0)

    # 第一次孔洞填充
    im_floodfill = img1.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img2 = img1 | im_floodfill_inv
    cv2.imwrite('image_fill.png', img2)
    # cv2.imshow('Image filling', img2)
    # cv2.waitKey(0)

    # 开启运算
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * 2 - 1, 6 * 2 - 1))  # 椭圆状的11x11掩膜
    img3 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel_open)
    # img3 = img3.astype(np.uint8)
    cv2.imwrite('opening circle.png', img3)
    # cv2.imshow('opening circle', img3)
    # cv2.waitKey(0)

    # 闭合运算
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * 2 - 1, 8 * 2 - 1))  # 椭圆状的15x15掩膜
    img4 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel_close)
    # 第二次孔洞填充
    im_floodfill = img4.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img4 = img4 | im_floodfill_inv
    cv2.imwrite('closing circle.png', img4)
    # cv2.imshow('closing circle', img4)
    # cv2.waitKey(0)

    # 在原图中提取番茄所在区域
    img5 = src
    for x in range(N):
        for y in range(M):
            if img4[x][y] == 0:
                img5[x][y][:] = 0
    img5 = img5.astype(np.uint8)
    cv2.imwrite('tomato area.png', img5)
    # cv2.imshow('tomato area', img5)
    # cv2.waitKey(0)

    # 提前到的番茄区域转为灰度图
    img6 = cv2.cvtColor(img5, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('gray tomato area.png', img6)
    # cv2.imshow('gray tomato area', img6)
    # cv2.waitKey(0)

    # 小波变换图像重构，增强图像边缘
    cA, (cH, cV, cD) = pywt.dwt2(img6, "haar")  # 低频分量、水平高频、垂直高频、对角线高频
    img7 = pywt.idwt2((0.6*cA, (4*cH, 4*cV, 4*cD)), "haar")   # 采用哈尔小波重构
    cv2.imwrite('haar.png', img7)
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

    # imgx = cv2.imread('canny.png')
    img9 = cv2.imread('canny.png')
    img10 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
    # img10 = img10.astype(np.uint8)
    # img10 = img10.astype(np.float32)

    # 霍夫圆检测，检测番茄的位置（番茄在平面的形状比较接近一个圆）
    circles0 = cv2.HoughCircles(img10, cv2.HOUGH_GRADIENT, 1,
                                40, param1=255, param2=9, minRadius=33, maxRadius=50)
    circles = circles0[0, :, :]
    circles = np.uint16(np.around(circles))
    for i in circles[:]:

        cv2.rectangle(SRC, (int(i[0]) - int(i[2]), int(i[1]) - int(i[2])),
                      (int(i[0]) + int(i[2]), int(i[1]) + int(i[2])), (255, 0, 0), 2)  # 画方框
        cv2.circle(SRC, (i[0], i[1]), 2, (0, 0, 0), 3)  # 画圆心
        cv2.putText(SRC, 'tomato', (i[0], i[1] + int(0.3 * i[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # cv2.circle(imgx, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
        # cv2.circle(imgx, (i[0], i[1]), 2, (255, 0, 255), 3)  # 画圆心
    # img10 = img10.astype(np.float32)
    cv2.imwrite('cache.jpg', SRC)
    # cv2.imwrite('holf.png', imgx)
    # cv2.imshow('result', SRC)
    # cv2.waitKey(0)


def orange_location(path):
    SRC = cv2.imread(path)
    src = SRC.astype(np.float32)
    # cv2.imshow('Source', SRC)
    # cv2.waitKey(0)

    [N, M] = SRC.shape[:2]  # 获取图片长宽
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    # 转为灰度图
    # gray = gray.astype(np.uint8)
    cv2.imwrite('gray_0.png', gray)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    # 对灰度图进行红绿色差法归一化，主要是显著突出原图黄色与绿色的差异，然后以灰度图的形式呈现
    for x in range(N):
        for y in range(M):
            gray[x][y] = NRG(x, y, src)

    gray = gray.astype(np.uint8)
    cv2.imwrite('gray_nyg.png', gray)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    _, img1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # Otsu法二值化
    cv2.imwrite('binary.png', img1)
    # cv2.imshow('OTSU', img1)
    # cv2.waitKey(0)

    # 第一次孔洞填充
    im_floodfill = img1.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img2 = img1 | im_floodfill_inv
    # cv2.imwrite('image_fill.png', img2)
    # cv2.imshow('Image filling', img2)
    # cv2.waitKey(0)

    # 开启运算
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * 2 - 1, 6 * 2 - 1))  # 椭圆状的11x11掩膜
    img3 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel_open)
    # img3 = img3.astype(np.uint8)
    cv2.imwrite('opening circle.png', img3)
    # cv2.imshow('opening circle', img3)
    # cv2.waitKey(0)

    # 闭合运算
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * 2 - 1, 8 * 2 - 1))  # 椭圆状的15x15掩膜
    img4 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel_close)
    # 第二次孔洞填充
    im_floodfill = img4.copy()
    mask = np.zeros((N + 2, M + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img4 = img4 | im_floodfill_inv
    cv2.imwrite('closing circle.png', img4)
    # cv2.imshow('closing circle', img4)
    # cv2.waitKey(0)

    # 在原图中提取番茄所在区域
    img5 = src
    for x in range(N):
        for y in range(M):
            if img4[x][y] == 0:
                img5[x][y][:] = 0
    img5 = img5.astype(np.uint8)
    # cv2.imwrite('tomato area.png', img5)
    # cv2.imshow('tomato area', img5)
    # cv2.waitKey(0)

    # 提前到的橙子区域转为灰度图
    img6 = cv2.cvtColor(img5, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite('gray tomato area.png', img6)
    # cv2.imshow('gray tomato area', img6)
    # cv2.waitKey(0)

    # 小波变换图像重构，增强图像边缘
    cA, (cH, cV, cD) = pywt.dwt2(img6, "haar")  # 低频分量、水平高频、垂直高频、对角线高频
    img7 = pywt.idwt2((cA, (cH, cV, cD)), "haar")   # 使用哈尔小波
    cv2.imwrite('haar.png', img7)
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

    # cA, (cH, cV, cD) = pywt.dwt2(img8, "haar")  # 低频分量、水平高频、垂直高频、对角线高频
    # imgx = pywt.idwt2((cA, (cH, cV, cD)), "haar")   # 使用哈尔小波
    # cv2.imwrite('the_haar.png', imgx)

    img9 = cv2.imread('canny.png')
    img10 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
    # img10 = img10.astype(np.uint8)
    # img10 = img10.astype(np.float32)

    # 霍夫圆检测，检测橙子的位置（橙子在平面的形状比较接近一个圆）
    circles0 = cv2.HoughCircles(img10, cv2.HOUGH_GRADIENT, 1,
                                40, param1=255, param2=9, minRadius=85, maxRadius=99)
    circles = circles0[0, :, :]
    circles = np.uint16(np.around(circles))
    signal = 0
    # print('circles:', circles)
    # print('len(circles):', len(circles))

    # print('circles:', len(circles))
    delete_index = []   # 删除索引：用于删除某些不符合要求的圆心
    for index, i in enumerate(circles[:]):
        if SRC[i[1]][i[0]][2] <= 100 and SRC[i[1]][i[0]][1] <= 100:  # 如果检测到的圆心不是黄色
            delete_index.append(index)
    #     print('index:', index, 'SRC[i[1]][i[0]][2]', SRC[i[1]][i[0]][2], 'SRC[i[1]][i[0]][1]', SRC[i[1]][i[0]][1])
    # print('delete_index', delete_index)

    if len(delete_index) > 0:
        circles = np.delete(circles, delete_index, 0)

    # print('circles:', len(circles))
    # 类似非极大值抑制算法，在两个极为接近的圆心中删除一个
    for i in range(len(circles)):
        for j in range(len(circles)):
            if i == j:
                continue
            threshold = math.sqrt(
                math.pow(int(circles[i][0]) - int(circles[j][0]), 2) +
                math.pow(int(circles[i][1]) - int(circles[j][1]), 2))
            # print('threshold:', threshold)
            # print(circles[i][0], circles[j][0], circles[i][1], circles[j][1])
            if threshold < int(circles[i][2]) / 2:
                circles = np.delete(circles, i, 0)
                signal = 1
                break
        if signal == 1:
            break

    # print('circles:', circles)
    # print('len(circles):', len(circles))
    for i in circles[:]:
        # cv2.circle(SRC, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
        cv2.rectangle(SRC, (int(i[0]) - int(i[2]), int(i[1]) - int(i[2])),
                      (int(i[0]) + int(i[2]), int(i[1]) + int(i[2])), (255, 0, 0), 2)  # 画方框
        cv2.circle(SRC, (i[0], i[1]), 2, (0, 0, 0), 3)  # 画圆心
        cv2.putText(SRC, 'orange', (i[0], i[1] + int(0.3 * i[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        # print('i[0]-i[2]:', np.array(i[0])-np.array(i[2]))
        # print('i[1]-i[2]:', np.array(i[1])-np.array(i[2]))
        # print('i[0]:', i[0])
        # print('i[1]:', i[1])
        # print('i[2]:', i[2])
    # img10 = img10.astype(np.float32)
    cv2.imwrite('cache.jpg', SRC)
    # cv2.imshow('result', SRC)
    # cv2.waitKey(0)


# 主要功能函数fruit_location()
def fruit_location(path):
    SRC = cv2.imread(path)
    [N, M] = SRC.shape[:2]
    threashold = 0
    for x in range(N):
        for y in range(M):
            threashold += SRC[x][y][2] / (M * N)
    # print(threashold)
    if threashold > 140:
        tomato_location(path)
    else:
        orange_location(path)


if __name__ == '__main__':
    path = 'orange.png'
    fruit_location(path)
