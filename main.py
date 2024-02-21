import numpy as np
import cv2
import imutils


def region_of_interest(img, vertices):  # define mask for area of interest
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def imgPipeline(image):

    height = image.shape[0]  # 1080
    width = image.shape[1]  # 1920

    minx0 = round(width * 0.2)
    miny0 = round(height * 0.1)
    maxx0 = round(width * 0.8)
    maxy0 = round(height * 0.9)

    region_of_interest_vertices = [
        (minx0, miny0),
        (maxx0, miny0),
        (maxx0, maxy0),
        (minx0, maxy0)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cannyed_image = cv2.Canny(blur, 200, 250, apertureSize=5)

    cropped_image = region_of_interest(
        cannyed_image,
        # threshold_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    # https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/

    contours, hierarchy = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours((contours, hierarchy))
    new = list(contours)
    new.sort(key=cv2.contourArea)

    if len(new) > 1:
        c1 = new[-1]
        c2 = new[-2]

        outline1 = cv2.approxPolyDP(c1, 4, False)
        cv2.drawContours(image, [outline1], -1, (0, 0, 255), 10)

        outline2 = cv2.approxPolyDP(c2, 4, False)
        cv2.drawContours(image, [outline2], -1, (0, 255, 255), 10)

        midline = []

        for pt1, pt2 in zip(outline1[:len(outline1)//2], outline2):
            mid_x = int((pt1[0][0] + pt2[0][0])/2)
            mid_y = int((pt1[0][1] + pt2[0][1])/2)
            midline.append([[mid_x, mid_y]])

        midline = np.array(midline, dtype=np.int32)

        cv2.polylines(image, [midline], False, (0, 255, 0), 10)

    cv2.rectangle(image, (minx0, miny0), (maxx0, maxy0), (255, 0, 0), 10)
    return image


# Open the camera stream for processing
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Camera Not Accessible, Try Agan")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Cannot receive frame. Assuming stream end. Process killed")
        break
    cv2.imshow('MiddleLine', imgPipeline(frame))
    if cv2.waitKey(1) == ord('q'):
        break
