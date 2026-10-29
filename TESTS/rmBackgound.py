import cv2
import numpy as np

bg = cv2.imread("./assets/bg.jpg")
image = cv2.imread("./assets/input/5.jpg")

# Subtract the background from the image
kSize = (5, 5)
sigma = 2
result = cv2.subtract(cv2.GaussianBlur(image, kSize, sigma), cv2.GaussianBlur(bg, kSize, sigma))
#cv2.imshow("Background Removed", result)
#cv2.imshow("Grayscale", cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))

#threshold
threshold = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 15, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("Threshold", threshold)

#floodfill: fill holes inside the foreground blob using a background flood from the corner
im_floodfill = threshold.copy()
h, w = im_floodfill.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
filled = threshold | im_floodfill_inv

#cv2.imshow("Floodfill", im_floodfill)
#cv2.imshow("Floodfill Inv", im_floodfill_inv)
cv2.imshow("Filled", filled)

num, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, 8)
print("Number of connected components:", num - 1)  # subtract 1 for the background label
print("Labels:", labels)
print("Stats:", stats)
print("Centroids:", centroids)

image_w_bboxes = image.copy()
for i in range(1, num):  # skip the background label
    x, y, w, h, area = stats[i]
    if(area < 400):  # filter out small components
        continue
    if(area > image.shape[0] * image.shape[1] * 0.8):  # filter out large components
        continue
    add = 10
    x -= add/2
    y -= add/2
    w += add
    h += add
    print(f"Component {i}: x={x}, y={y}, w={w}, h={h}, area={area}")
    cv2.rectangle(image_w_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image_w_bboxes, f"ID {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Image with Bounding Boxes", image_w_bboxes)

cv2.waitKey(0)
cv2.destroyAllWindows()