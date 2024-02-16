import cv2

img_1 = cv2.imread("image_test/1636738315284889400.png", cv2.IMREAD_COLOR)

cv2.imshow("img_1_org", img_1)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)

cv2.imshow("img_1_b4", img_1)
img_1[:,:,2] = 255

cv2.imshow("img_1_hsv_after", img_1)

img_1 = cv2.cvtColor(img_1, cv2.COLOR_HSV2BGR)

cv2.imshow("img_1__bgr_after", img_1)
