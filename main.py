import numpy as np
import cv2
import imutils


def region_of_interest(img, vertices):  # define mask for area of interest
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def imgPipeline(image, thresh):

    height = image.shape[0]  # 1080
    width = image.shape[1]  # 1920

    minx0 = round(width * 0.1)
    miny0 = round(height * 0.1)
    maxx0 = round(width * 0.9)
    maxy0 = round(height * 0.9)

    region_of_interest_vertices = [
        (minx0, miny0),
        (maxx0, miny0),
        (maxx0, maxy0),
        (minx0, maxy0)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    cannyed_image = cv2.Canny(blur, threshold1=thresh, threshold2=thresh * 2, apertureSize=3)

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
        cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

        outline1 = cv2.approxPolyDP(c1, 4, False)
        cv2.drawContours(image, [outline1], -1, (0, 0, 255), 3)

        outline2 = cv2.approxPolyDP(c2, 4, False)
        cv2.drawContours(image, [outline2], -1, (0, 255, 255), 3)

        midline = []

        for pt1, pt2 in zip(outline1, outline2):
            mid_x = int((pt1[0][0] + pt2[0][0])/2)
            mid_y = int((pt1[0][1] + pt2[0][1])/2)
            midline.append([[mid_x, mid_y]])

        midline = np.array(midline, dtype=np.int32)

        cv2.polylines(image, [midline], False, (0, 255, 0), 10)

        # mid_mid = (int((outline1[int(len1/2)][0][0] + outline2[int(len2/2)][0][0])/2),
        #            int((outline1[int(len1/2)][0][1] + outline2[int(len2/2)][0][1])/2))
        # cv2.circle(image, mid_mid, 10, (100, 100, 100), -1)

        # s_diffx = outline1[0][0][0] - start_mid[0]
        # s_diffy = outline1[0][0][1] - start_mid[1]
        # e_diffx = outline1[len1][0][0] - end_mid[0]
        # e_diffy = outline1[len1][0][1] - end_mid[1]

        # extLeft1 = tuple(c1[c1[:, :, 0].argmin()][0])
        # extRight1 = tuple(c1[c1[:, :, 0].argmax()][0])
        # extTop1 = tuple(c1[c1[:, :, 1].argmin()][0])
        # extBot1 = tuple(c1[c1[:, :, 1].argmax()][0])
        #
        # extLeft2 = tuple(c2[c2[:, :, 0].argmin()][0])
        # extRight2 = tuple(c2[c2[:, :, 0].argmax()][0])
        # extTop2 = tuple(c2[c2[:, :, 1].argmin()][0])
        # extBot2 = tuple(c2[c2[:, :, 1].argmax()][0])
        #
        # print("l", extLeft1)
        # print("r", extRight1)
        # print("t", extTop1)
        # print("b", extBot1)
        #
        # l_diff = int((extLeft1[0] - extLeft2[0])/2)
        # r_diff = int((extRight1[0] - extRight2[0])/2)
        # t_diff = int((extTop1[0] - extTop2[0])/2)
        # b_diff = int((extBot1[0] - extBot2[0])/2)
        #
        # mainx = [l_diff, r_diff]
        # mainy = [t_diff, b_diff]
        #
        # print(max(mainx))
        # print(max(mainy))
        #
        # midline = outline2 + [max(mainx), -max(mainy)]
        #
        # new = scale_contour(midline, 1.2)
        #
        # cv2.drawContours(image, [new], -1, (0, 255, 0), 10)

        # eq1 = quad_equation(start_mid[0], start_mid[1])
        # eq2 = quad_equation(end_mid[0], end_mid[1])
        # eq3 = quad_equation(mid_mid[0], mid_mid[1])
        #
        # print(sp.solve([eq1, eq2, eq3]))



        # #find vertex (aka point with equal distance from either point)
        #
        # vertex = []
        #
        # s_e_dist = math.dist(list(start_mid), list(end_mid))
        # print("se", s_e_dist)
        # s_m_dist = math.dist(list(start_mid), list(mid_mid))
        # print("sm", s_m_dist)
        # e_m_dist = math.dist(list(end_mid), list(mid_mid))
        # print("em", e_m_dist)
        #
        # if abs(s_e_dist - s_m_dist) < 120:
        #     print("start")
        #     vertex = start_mid
        # elif abs(s_e_dist - e_m_dist) < 120:
        #     print("end")
        #     vertex = end_mid
        # else:
        #     print("mid")
        #     vertex = mid_mid
        #
        # endpoints = [start_mid, end_mid, mid_mid]
        # endpoints.remove(vertex)

        # print("s", start_mid)
        # print("e", end_mid)
        # print("m", mid_mid)
        #
        # pts = np.array([[start_mid[0], start_mid[1]],
        #                 [mid_mid[0], mid_mid[1]],
        #                 [end_mid[0], end_mid[1]]], np.int32)
        #
        # coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
        # poly = np.poly1d(coeffs)
        #
        # yarr = np.arange(start_mid[1], end_mid[1])
        # xarr = poly(yarr)
        #
        # parab_pts = np.array([xarr, yarr], dtype=np.int32).T
        # cv2.polylines(image, [parab_pts], False, (255, 0, 0), 3)

    cv2.rectangle(image, (minx0, miny0), (maxx0, maxy0), (255, 0, 0), 10)
    return image


# img = cv2.imread("curved.png")
# cv2.imshow("source", imgPipeline(img, 100))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Open the camera stream for processinq
camera = cv2.VideoCapture(0)  # 0 for built-in or default camera and 1 for second camera
if not camera.isOpened():
    print("Camera Not Accessible, Try Agan")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Cannot receive frame. Assuming stream end. Process killed")
        break
    cv2.imshow('Source View', imgPipeline(frame, 150))
    if cv2.waitKey(1) == ord('q'):
        break
