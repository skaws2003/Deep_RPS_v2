import cv2
import numpy as np


range1_d = np.array([0,50,0])
range1_u = np.array([20,170,255])
range2_d = np.array([170,50,0])
range2_u = np.array([180,170,255])


def to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, range1_d, range1_u)
    mask2 = cv2.inRange(hsv, range2_d, range2_u)
    mask = cv2.bitwise_or(mask1,mask2)
    return mask


print("Processing Start...")
for i in range(1,802):
    cv2.imwrite("D:/RPSDATAv2/rock (%d).bmp"%i ,to_hsv(cv2.imread("D:/RPSDATAv2_raw/rock (%d).jpg"%i)))
    cv2.imwrite("D:/RPSDATAv2/paper (%d).bmp"%i ,to_hsv(cv2.imread("D:/RPSDATAv2_raw/paper (%d).jpg"%i)))
    cv2.imwrite("D:/RPSDATAv2/scissors (%d).bmp"%i ,to_hsv(cv2.imread("D:/RPSDATAv2_raw/scissors (%d).jpg"%i)))
    if i%50 ==0:
        print("Processing done(%d/800)"%i)
    
