import cv2
image = cv2.imread("./Circles2.jpg")
cv2.imshow("Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()