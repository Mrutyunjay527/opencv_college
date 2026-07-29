import cv2
import numpy as np
import matplotlib.pyplot as plt

# to import the image and read it
img = cv2.imread("ribcage.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#to add brightness
bright = cv2.add(img, np.ones(img.shape, dtype=np.uint8) *50)
dark = cv2.subtract(img, np.ones(img.shape, dtype=np.uint8) *50) 

# to flip the image
h_flip = cv2.flip(img, 1) # horizontal flip
v_flip = cv2.flip(img, 0) # verticle flip

# color channels
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

# gray scaling
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# negative of the image 
negative = 255 - img

# display all image
plt.figure(figsize=(18,12))

# row-1 brightness
plt.subplot(3, 4, 1)
plt.imshow(img);         plt.title('Original');      plt.axis('off')
plt.subplot(3, 4, 2)
plt.imshow(bright);        plt.title('Brighter +50');  plt.axis('off')
plt.subplot(3, 4, 3)
plt.imshow(dark);          plt.title('Darker -50');    plt.axis('off')

#row-2 flip
plt.subplot(3, 4, 5)
plt.imshow(h_flip);        plt.title('H-Flip');        plt.axis('off')
plt.subplot(3, 4, 6)
plt.imshow(v_flip);        plt.title('V-Flip');        plt.axis('off')

#row-2 color channels
plt.subplot(3, 4, 7)
plt.imshow(red,   cmap='Reds');   plt.title('Red');   plt.axis('off')
plt.subplot(3, 4, 8)
plt.imshow(green, cmap='Greens'); plt.title('Green'); plt.axis('off')

# Row 3
plt.subplot(3, 4, 9)
plt.imshow(blue,  cmap='Blues');  plt.title('Blue');  plt.axis('off')
plt.subplot(3, 4, 10)
plt.imshow(gray,  cmap='gray');   plt.title('Gray');  plt.axis('off')
plt.subplot(3, 4, 11)
plt.imshow(negative);  plt.title('Negative'); plt.axis('off')

# bright image 
plt.tight_layout()
plt.show()

plt.subplot(1, 3, 2)
plt.imshow(bright)
plt.title('Brighter Image (+50)')
plt.axis('off')

#dark image
plt.subplot(1, 3, 3)
plt.imshow(dark)
plt.title('Darker Image (-50)')
plt.axis('off')


plt.show()