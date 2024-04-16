import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read in the image
image = cv2.imread('../data/clustering.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# The below line of code defines the criteria for the algorithm to stop running,
# which will happen if 100 iterations are run or the epsilon (which is the required accuracy)
# becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Then perform k-means clustering with the number of clusters defined as 3
# Also, random centres are initially chosen for k-means clustering
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# Reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

# Display the segmented image
plt.imshow(segmented_image)
plt.axis('off')
plt.show()

# Save the segmented image
cv2.imwrite('segmented_image.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
