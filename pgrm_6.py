#import the dependencies
import cv2
import numpy as np
image = cv2.imread("pneumonia.png")
image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# performing the edge detetcion
canny_output = cv2.Canny(image, 80, 150)

cv2.imshow('Canny', canny_output)
cv2.waitKey()