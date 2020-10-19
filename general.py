import cv2
import numpy as np

#filling holes in objects which might have appeared after conversoin to black and white
def fill_holes(bw_image):
    im_floodfill = bw_image.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = bw_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)

    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#    Combine the two images to get the foreground.
    im_out = bw_image | im_floodfill_inv
    return im_out

# finding background color of the image
def find_background_color(image):
    colors_count = {}
    (channel_b, channel_g, channel_r) = cv2.split(image)
    # Flattens the 2D single channel array so as to make it easier to iterate over it
    channel_b = channel_b.flatten()
    channel_g = channel_g.flatten()
    channel_r = channel_r.flatten()

    for i in range(len(channel_b)):
        RGB = str(channel_r[i]) + "," + str(channel_g[i]) + "," + str(channel_b[i])
        if RGB in colors_count:
            colors_count[RGB] += 1
        else:
            colors_count[RGB] = 1
    return sorted(colors_count, key=colors_count.__getitem__)[colors_count.__len__()-1]

def count_avg_perimeter(cnts):
    sum = 0  # the total perimeter
    count = 0
    # calculating the average perimeter:
    for cnt in cnts:
        sum = sum + cv2.contourArea(cnt)
        count = count + 1
    avg = sum / count  # average perimeter for all objects
    return avg

def divide_cnts_by_size(cnts, image):
    avg = count_avg_perimeter(cnts)
    xcnts=[]  # larger than the average perimeter
    ycnts = []  # smaller than the average perimeter
    # divide the circles into two groups: larger and smaller than the average perimeter:
    for cnt in cnts:
        if cv2.contourArea(cnt) > avg:
            xcnts.append(cnt)
        else:
            ycnts.append(cnt)

    cv2.drawContours(image, xcnts, -1, (0, 0, 255), 2)
    cv2.drawContours(image, ycnts, -1, (255, 0, 0), 2)

def add_padding(image):
    colors = find_background_color(image)
    color = colors.split(',')
    image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, (int(color[0]), int(color[1]), int(color[2])))
    return image

def morphology_operations(image):
    img2 = image
    erosionimg = cv2.erode(img2, kernel, iterations = 1)
    openimg = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    rezimg = openimg - erosionimg
    grayImage = cv2.cvtColor(rezimg, cv2.COLOR_BGR2GRAY)
    closeimg = cv2.morphologyEx(grayImage, cv2.MORPH_CLOSE, kernel, iterations=11)
    return closeimg

image = cv2.imread("./coins2.jpg")
kernel = np.ones((5, 5), np.uint8)
# getting rid of the border
height = image.shape[0]
width = image.shape[1]
image = image[5: height - 5, 5: width - 5]
# adding a padding to the image the color of the background
image = add_padding(image)
# morphology operation
closeimg = morphology_operations(image)
(thresh, blackAndWhiteImage) = cv2.threshold(closeimg, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
filled_holes = fill_holes(blackAndWhiteImage)
cnts = cv2.findContours(filled_holes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
divide_cnts_by_size(cnts, image)
image = image[5: height - 5, 5: width - 5]
cv2.imshow("coins", image)
cv2.waitKey(0)
cv2.destroyAllWindows()