import cv2

# Load the images
image1 = cv2.imread(r"C:\Users\arif1\OneDrive\Pictures\Screenshots\Screenshot (69).png")
image2 = cv2.imread(r"C:\Users\arif1\OneDrive\Pictures\Screenshots\Screenshot (70).png")

# Check if the images were loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both of the images could not be loaded.")
    exit()

# Create a list of images to stitch
images = [image1, image2]

# Create a Stitcher object with SIFT as the method
stitcher = cv2.createStitcher() if cv2.__version__.startswith('3') else cv2.createStitcherSift()

# Attempt to stitch the images
result = stitcher.stitch(images)

# Check if the stitching was successful
if result[0] == cv2.Stitcher_OK:
    panorama = result[1]

    # Display the panorama
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the panorama to a file
    cv2.imwrite('panorama.jpg', panorama)
else:
    print("Error during stitching.")
