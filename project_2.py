import cv2
import numpy as np

img = cv2.imread("highland.png")

if img is None:
    print("image not found")
    exit()
    
    
# image dimensions
rows, cols = img.shape[:2]


# Translations
tx, ty = 80, 50
translation_matrix = np.float32([[1, 0, tx],
                                 [0, 1, ty]])

translated = cv2.warpAffine(img, translation_matrix, (cols, rows))

# rotation
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)

rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))

# Scaling
scaled = cv2.resize(img, None, fx=1.5, fy=1.5,
                    interpolation=cv2.INTER_LINEAR)

# flipping
flip_horizontal = cv2.flip(img, 1)
flip_vertical = cv2.flip(img, 0)
flip_both = cv2.flip(img, -1)

# affine Transformation
pts1 = np.float32([[50,50],
                   [200,50],
                   [50,200]])

pts2 = np.float32([[10,100],
                   [200,50],
                   [100,250]])

affine_matrix = cv2.getAffineTransform(pts1, pts2)

affine = cv2.warpAffine(img, affine_matrix, (cols, rows))

# perspective transformation
pts1 = np.float32([[50,50],
                   [300,50],
                   [50,300],
                   [300,300]])

pts2 = np.float32([[0,0],
                   [300,0],
                   [100,300],
                   [300,300]])

perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

perspective = cv2.warpPerspective(img,
                                  perspective_matrix,
                                  (cols, rows))

# display results
cv2.imshow("Original", img)
cv2.imshow("Translated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Scaled", scaled)
cv2.imshow("Flip Horizontal", flip_horizontal)
cv2.imshow("Flip Vertical", flip_vertical)
cv2.imshow("Affine", affine)
cv2.imshow("Perspective", perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()