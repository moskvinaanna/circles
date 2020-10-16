import cv2
import numpy as np
image = cv2.imread("./Circles2.jpg")
height = image.shape[0]
width = image.shape[1]
image = image[5: height - 5, 5: width - 5]
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_inverse = cv2.bitwise_not(blackAndWhiteImage)
cnts = cv2.findContours(thresh_inverse, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

sum = 0
count = 0
xcnts=[]
ycnts = []
for cnt in cnts:
    sum = sum + cv2.contourArea(cnt)
    count = count + 1
avg = sum / count
cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
for cnt in cnts:
    if cv2.contourArea(cnt) > avg:
        xcnts.append(cnt)
    else:
        ycnts.append(cnt)
cv2.drawContours(image, xcnts, -1, (255, 0, 0), cv2.FILLED)
cv2.drawContours(image, ycnts, -1, (0, 0, 255), cv2.FILLED)
print(avg)
cv2.imshow("Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()