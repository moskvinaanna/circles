import cv2
import numpy as np
image = cv2.imread("./Circles2.jpg")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_inverse = cv2.bitwise_not(grayImage)
cnts = cv2.findContours(thresh_inverse, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

s1 = 5000
xcnts=[]
for cnt in cnts:
    if cv2.contourArea(cnt):
        xcnts.append(cnt)
cv2.drawContours(image, xcnts, -1, (0, 255, 0), 1)
print(len(xcnts))
cv2.imshow("Circles", thresh_inverse)
cv2.waitKey(0)
cv2.destroyAllWindows()