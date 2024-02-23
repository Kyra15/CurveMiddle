# import the filter file functions
from filters import *


# process a frame given by applying functions
def process(image):
    # get the coordinates for the mask
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

    # apply the filters onto the given frame
    filtered_img = filters(image)

    # apply the mask onto the given frame
    cropped_image = region_of_interest(
        filtered_img,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    # find the contours on the image
    contours, hierarchy = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours((contours, hierarchy))

    # sort the list of contours by the contour area
    new = list(contours)
    new.sort(key=cv2.contourArea)

    # if there are at least 2 contours that have been detected
    if len(new) > 1:
        # get the 2 largest contours
        c1 = new[-1]
        c2 = new[-2]

        # fit polylines to each contour
        outline1 = cv2.approxPolyDP(c1, 4, False)
        cv2.drawContours(image, [outline1], -1, (0, 0, 255), 10)

        outline2 = cv2.approxPolyDP(c2, 4, False)
        cv2.drawContours(image, [outline2], -1, (0, 255, 255), 10)

        # draw a midline by going through the polyline and averaging each x and y coordinate
        # append this averaged coordinate to a list and turn that list into a numpy array
        midline = []

        for pt1, pt2 in zip(outline1[:len(outline1)//2], outline2):
            mid_x = int((pt1[0][0] + pt2[0][0])/2)
            mid_y = int((pt1[0][1] + pt2[0][1])/2)
            midline.append([[mid_x, mid_y]])

        midline = np.array(midline, dtype=np.int32)

        # draw a polyline from the numpy array onto the frame
        cv2.polylines(image, [midline], False, (0, 255, 0), 20)

    # draw a rectangle on the frame to show the mask and return the final frame
    cv2.rectangle(image, (minx0, miny0), (maxx0, maxy0), (255, 0, 0), 10)
    return image
