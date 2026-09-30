import cv2
import matplotlib.pyplot as plt


img= cv2.imread(r"bird.png")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Step 3: Apply Threshold
# Rule: if pixel > 127 → 255 (white)
#       if pixel ≤ 127 → 0   (black)
T = 127
ret, result = cv2.threshold(gray, T, 255,
                             cv2.THRESH_BINARY)

# Step 4: Show all 3 images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Step 1: Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Step 2: Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result, cmap='gray')
plt.title(f'Step 3: Threshold Result (T={T})')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Threshold value used : {T}")
print(f"Pixels in result     : only 0 and 255")