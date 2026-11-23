import cv2
import numpy as np

# Load the images
image1 = cv2.imread(r"C:\Users\arif1\OneDrive\Pictures\Screenshots\Screenshot (69).png")
image2 = cv2.imread(r"C:\Users\arif1\OneDrive\Pictures\Screenshots\Screenshot (70).png")

# Check if the images were loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both of the images could not be loaded.")
    exit()

# Convert the images to grayscale for feature matching
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize feature detector and descriptor
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match the keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
matched_img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Resize the window to fit the image
cv2.namedWindow('Matched Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matched Image', matched_img.shape[1], matched_img.shape[0])

# Display the matched image
cv2.imshow('Matched Image', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Proceed with the rest of the stitching code as before
# ...
