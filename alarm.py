import cv2
import numpy as np
src = cv2.imread("./alarm/1.jpg")
out_1 = './results/alarm/1.jpg'
out_2 = './results/alarm/2.jpg'
out_3 = './results/alarm/3.jpg'

import cv2
import numpy as np
 
 
if __name__ == "__main__":
    img = cv2.imread('./alarm/1.jpg')
    # red_RGB = np.array([217, 33, 13])
    # green_RGB = np.array([0, 255, 0])
    # # print(img.shape)
    # mask = np.zeros((img.shape[0], img.shape[1], 3))
    # for w in range(img.shape[0]):
    #     for h in range(img.shape[1]):
    #         r, g, b = img[w, h]
    #         # print(r, g, b)
    #         # if r == 217 and g == 33 and b == 13:
    #         #     mask[w, h] = img[w, h]
    #         if r == 0 and g == 255 and b == 0:
    #             mask[w, h] = img[w, h]

    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lower1 = np.array([150, 0, 0])
    upper1 = np.array([255, 100, 100])
    mask1 = cv2.inRange(grid_RGB, lower1, upper1)       
    cv2.imwrite(out_1, mask1) 

    lower2 = np.array([0,150,0])
    upper2 = np.array([100,255,100])
    mask2 = cv2.inRange(grid_RGB, lower2, upper2)
    cv2.imwrite(out_2, mask2)

    mask3 = cv2.bitwise_or(mask1, mask2)
    cv2.imwrite(out_3, mask3)
 
    # # 从RGB色彩空间转换到HSV色彩空间
    # grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
 
    # # H、S、V范围一：
    # lower1 = np.array([0,43,46])
    # upper1 = np.array([10,255,255])
    # mask1 = cv2.inRange(grid_HSV, lower1, upper1)       
    # # res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)
    # cv2.imwrite(out_1, mask1) 
 
    # # H、S、V范围二：
    # lower2 = np.array([156,43,46])
    # upper2 = np.array([180,255,255])
    # mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    # cv2.imwrite(out_2, mask1)
    # # res2 = cv2.bitwise_and(grid_RGB,grid_RGB, mask=mask2)
 
    # # 将两个二值图像结果 相加
    # mask3 = mask1 + mask2
    # # mask3 = cv2.bitwise_and(mask1, mask2)
    # cv2.imwrite(out_3, mask1)

    # lower1 = np.array([60,43,46])
    # upper1 = np.array([65,255,255])
    # mask1 = cv2.inRange(grid_HSV, lower1, upper1)       
    # cv2.imwrite(out_1, mask1) 

# cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input", src)
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# low_hsv = np.array([0,43,46])
# high_hsv = np.array([10,255,255])
# mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
# cv2.imwrite(out, mask) 
# cv2.imshow("test",mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
