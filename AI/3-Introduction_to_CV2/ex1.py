import cv2

img = cv2.imread("sample1.png")
cv2.imshow("sample1",img)
cv2.imwrite("sample1Copy.png",img)
cv2.waitKey(0)  # Wait for infinite time before continuing
cv2.destroyAllWindows()
