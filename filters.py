# import necessary libraries
import cv2
import numpy as np


# applies a grayscale filter, gaussian blur, and canny image filter on a frame given and returns the final frame
def filters(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (7, 7), 0)
    cv2.imshow("blur", blur)
    cannyed_image = cv2.Canny(blur, 250, 300, apertureSize=5)
    cv2.imshow("cann", cannyed_image)
    return cannyed_image


# function that created a mask on an image given the coordinates of the mask
# and returns the masked image
def masking(img, vertices):  # define mask for area of interest
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
