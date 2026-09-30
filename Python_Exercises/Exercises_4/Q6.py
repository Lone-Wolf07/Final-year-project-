import os
import cv2
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

image1_path = input("Enter first image path: ")
image2_path = input("Enter second image path: ")

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda m: m.distance)

points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

height = img1.shape[0]
width = img1.shape[1] + img2.shape[1]
panorama = cv2.warpPerspective(img2, homography, (width, height))
panorama[0:img1.shape[0], 0:img1.shape[1]] = img1

cv2.imwrite(os.path.join(output_dir, "panorama.jpg"), panorama)
print("Panorama saved")
