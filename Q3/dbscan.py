import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN

# Read in the image
image = cv2.imread('../data/clustering.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# Define DBSCAN parameters
eps = 10  # maximum distance between two samples for one to be considered as in the neighborhood of the other
min_samples = 100  # the number of samples in a neighborhood for a point to be considered as a core point

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(pixel_vals)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Convert data into 8-bit values
centers = np.uint8([[np.mean(pixel_vals[labels == i], axis=0)] for i in range(n_clusters_)])

# Map each pixel to its cluster center
segmented_data = centers[labels.flatten()]

# Reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

# Display the segmented image
plt.imshow(segmented_image)
plt.axis('off')
plt.show()

# Save the segmented image
cv2.imwrite('segmented_image_dbscan.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
