import cv2
import numpy as np
image = cv2.imread("./coins.jpg")
height = image.shape[0]
width = image.shape[1]
kernel = np.ones((20, 20), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
image = image[5: height - 5, 5: width - 5]

# morphology operation
img2 = image
erosionimg = cv2.erode(img2, kernel, iterations=1)
openimg = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
rezimg = openimg - erosionimg
grayImage = cv2.cvtColor(rezimg, cv2.COLOR_BGR2GRAY)
closeimg = cv2.morphologyEx(grayImage, cv2.MORPH_CLOSE, kernel2)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_inverse = cv2.bitwise_not(blackAndWhiteImage)
cnts2 = cv2.findContours(thresh_inverse, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

sum = 0  # the total perimeter
count = 0
xcnts=[]  # larger than the average perimeter
ycnts = []  # smaller than the average perimeter
# calculating the average perimeter:
for cnt in cnts2:
    sum = sum + cv2.contourArea(cnt)
    count = count + 1
avg = sum / count  # average perimeter for all objects

# divide the circles into two groups: larger and smaller than the average perimeter:
for cnt in cnts2:
    if cv2.contourArea(cnt) > avg:
        xcnts.append(cnt)
    else:
        ycnts.append(cnt)

cv2.drawContours(rezimg, xcnts, -1, (255, 0, 0), 2)
cv2.drawContours(rezimg, ycnts, -1, (0, 0, 255), 2)
cv2.imshow("Circles", rezimg)
print(avg)
cv2.waitKey(0)
cv2.destroyAllWindows()