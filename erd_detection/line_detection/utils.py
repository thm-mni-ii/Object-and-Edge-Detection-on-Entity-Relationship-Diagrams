import math
import os

import cv2 as cv
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.morphology import dilation, square


def get_paths():
    absolute_path = os.path.abspath(__file__)
    file_directory = os.path.dirname(absolute_path)
    input_folder = os.path.join(file_directory, "input")
    output_folder = os.path.join(file_directory, "output")
    dataset_folder = os.path.join(input_folder, "")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return input_folder, output_folder, dataset_folder


def load_images(input_folder):
    image_paths = {}

    file_list = os.listdir(input_folder)

    image_extensions = [".png", ".jpg"]
    for file in file_list:
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_name = os.path.splitext(file)[0]
            image_path = os.path.join(input_folder, file)
            image_paths[image_name] = image_path

    return image_paths


def get_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours


def create_matrix(matrix_size, center_point):
    # Create an x*x matrix centered around the point at the given index
    matrix = [[0] * matrix_size for _ in range(matrix_size)]
    start_x = center_point[0] - (matrix_size - 1) // 2
    start_y = center_point[1] - (matrix_size - 1) // 2
    for j in range(matrix_size):
        for k in range(matrix_size):
            matrix[j][k] = (start_x + j, start_y + k)
    return matrix


def find_crossings(cnts, img, matrix_size):
    all_crs = []
    for cnt in cnts:
        cnt = cnt.astype(np.int32)
        cnt = cnt.reshape((-1, 1, 2))
        cnt = [tuple(sublist[0]) for sublist in cnt]
        previous_matrix = []
        # Dictionary to check how many times a pixel has been crossed
        point_crossed_counter = {}
        single_point_crossed_checker = {}
        single_points_to_remove = []
        # Iterate over each point in the contour
        for i in range(len(cnt)):
            found = False
            center_point = cnt[i]
            # Create a 3x3 matrix
            matrix = create_matrix(matrix_size, center_point)
            # Check if any point in matrix was part of previous matrix
            for row in matrix:
                for point in row:
                    if point in previous_matrix:
                        found = True
            # If True, go to next point and if point was not part of previous matrix, add point to single point dict
            if found:
                if center_point not in previous_matrix:
                    single_point_crossed_checker[center_point] = 1
                continue

            # Transform current matrix to list
            flattened_matrix = []
            for a in range(matrix_size):
                for b in range(matrix_size):
                    flattened_matrix.append((matrix[a][b][0], matrix[a][b][1]))
            # Iterate over all previous single points and prepare to remove them
            for sgl_pnt_crossed in single_point_crossed_checker:
                single_points_to_remove.append(sgl_pnt_crossed)
                # If a single point is not part of the current matrix, add 1 to the crossing count
                # The other points will get counted later
                if sgl_pnt_crossed not in flattened_matrix:
                    if sgl_pnt_crossed in point_crossed_counter:
                        point_crossed_counter[sgl_pnt_crossed] += 1
                    else:
                        point_crossed_counter[sgl_pnt_crossed] = 1
            # Remove single point crossings
            if len(single_points_to_remove) > 0:
                for sgl_point in single_points_to_remove:
                    if sgl_point in single_point_crossed_checker:
                        single_point_crossed_checker.pop(sgl_point)
            # Add points in matrix to point crossed dictionary
            for row in matrix:
                for point in row:
                    x = point[0]
                    y = point[1]
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        if img[y, x] == 255:
                            if point in point_crossed_counter:
                                point_crossed_counter[point] += 1
                            else:
                                point_crossed_counter[point] = 1
            # Clear previous matrix and create a new one
            previous_matrix.clear()
            for row in matrix:
                for point in row:
                    previous_matrix.append(point)
        # Save crossed points for every contour
        if any(count > 2 for count in point_crossed_counter.values()):
            for point, count in point_crossed_counter.items():
                if count > 2:
                    result_entry = {"contour": cnt, "point": point, "crossings": count}

            all_crs.append(result_entry)
    return all_crs


def calculate_centroid(img_thinned, all_crossings_with_cont, group):
    points_list = []
    for idx in group:
        contour = [c for c in all_crossings_with_cont[idx]["contour"]]
        points = [p for p in all_crossings_with_cont[idx]["point"]]
        points_list.append(points)
    if len(points_list) != 0:
        points_array = np.array(points_list)
        centroid = np.mean(points_array, axis=0)
        centroid = tuple(map(int, centroid))
        roi_size = 3
        x, y = centroid
        roi = img_thinned[
            max(0, y - roi_size) : min(img_thinned.shape[0], y + roi_size),
            max(0, x - roi_size) : min(img_thinned.shape[1], x + roi_size),
        ]
        white_pixel_count = cv.countNonZero(roi)
        # todo other method: Check if more than 2 points touch outside edge of roi.
        # todo other method: Check direction of lines?
        contour_points_count = 0
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                px, py = x - roi_size + j, y - roi_size + i
                if (px, py) in contour:
                    contour_points_count += 1
        if white_pixel_count > 7 and contour_points_count > 7:
            # debug imgs
            # print("Valid Intersection")
            # print(f'white_pixel_count: {white_pixel_count} {x,y}')
            # cv.imwrite(f"1valid:{x,y}.png", roi)
            return centroid
        # else:
        #     print("False Positive Intersection")
        #     print(f'white_pixel_count: {white_pixel_count} {x,y}')
        #     cv.imwrite(f"2invalid:{x,y}.png", roi)


def group_and_calculate_centroids(
    all_points, img_thinned, all_crossings_with_cont, distance_threshold=10
):
    all_points_array = [
        entry["point"] for entry in all_crossings_with_cont if entry is not None
    ]
    all_points_array = np.array(all_points_array)
    # Calculate pairwise distances between all points
    distances = cdist(all_points_array, all_points_array)
    point_groups = []
    for i, point in enumerate(all_points):
        added_to_group = False
        for group in point_groups:
            if any(dist < distance_threshold for dist in distances[group, i]):
                group.append(i)
                added_to_group = True
                break
        if not added_to_group:
            point_groups.append([i])
    # Calculate the centroid for each group
    centroids = [
        calculate_centroid(img_thinned, all_crossings_with_cont, group)
        for group in point_groups
    ]
    return centroids


def split_contour(cnt, inters, img):
    # print(cnt)
    cnt = [tuple(sublist[0]) for sublist in cnt]
    lines = []
    line1 = []
    line2 = []
    is_line1 = True
    counter = 10
    inters_size = 5
    start_x = inters[0] - (inters_size - 1) // 2
    start_y = inters[1] - (inters_size - 1) // 2
    inters_area = [[0] * inters_size for _ in range(inters_size)]
    for j in range(inters_size):
        for k in range(inters_size):
            inters_area[j][k] = (start_x + j, start_y + k)
    flattened_inters_area = []
    for a in range(inters_size):
        for b in range(inters_size):
            flattened_inters_area.append((inters_area[a][b][0], inters_area[a][b][1]))
    for point in cnt:
        if counter > 15:
            # Toggle between line1 and line2 if there's an intersection
            if point in flattened_inters_area:
                is_line1 = not is_line1
                counter = 0
        if is_line1:
            line1.append(point)
        else:
            line2.append(point)
        counter += 1
    line1_img = np.zeros_like(img)
    line2_img = np.zeros_like(img)
    line1 = np.array(line1)
    cv.polylines(line1_img, [line1], False, 255, thickness=2)
    # todo change name to contour number/name and image name
    # cv.imwrite(f"output/{counter}line1_img.jpg", line1_img)
    if len(line2) > 0:
        line2 = np.array(line2)
        cv.polylines(line2_img, [line2], False, 255, thickness=2)
        # cv.imwrite(f"output/{counter}line2_img.jpg", line2_img)
    # print(len(line1))
    # print(len(line2))
    # Convert the lines to numpy arrays
    line1 = np.array(line1)
    line2 = np.array(line2)
    lines.append(line1)
    lines.append(line2)
    return lines


def filter_crossings(all_crossings):
    all_crossings_filtered = []
    for index, crossings in enumerate(all_crossings):
        if crossings and any(count > 2 for count in crossings.values()):
            # print(f"Contour {index + 1} has more than 2 crossing:")
            for point, count in crossings.items():
                if count > 2:
                    # print(f"  Point {point}: {count} crossings")
                    all_crossings_filtered.append(point)
    return all_crossings_filtered


def split_all_contours(contours, centroids, img_thinned):
    contours = [contour for contour in contours if len(contour) > 0]
    for centroid in centroids:
        contour_index = -1
        for i, contour in enumerate(contours):
            if len(contour.shape) > 2:
                result = cv.pointPolygonTest(contour, centroid, True)
                if result >= -1:
                    contour_index = i
                    break
        # process point if a contour containing point is found
        if contour_index != -1:
            contours[contour_index] = contour.reshape((-1, 1, 2))
            newlines = split_contour(contours[contour_index], centroid, img_thinned)
            contours.pop(contour_index)
            newlines[0] = newlines[0].reshape((-1, 1, 2))
            newlines[1] = newlines[1].reshape((-1, 1, 2))
            contours.append(newlines[0])
            contours.append(newlines[1])
    contours = tuple(contours)
    return contours


# taken from table detection
def sort_contours(cnts, method="left-to-right"):
    """
    Sorts the found contours. Taken from Table

    :param cnts:
    :param method:
    :return: returns the list of sorted contours and bounding boxes
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )
    return cnts, boundingBoxes


def get_cnts_array(cnts):
    cnts_array = []
    for c in cnts:
        cnts_array.append(np.array(c, dtype=np.int32))
    return cnts_array


def canny(img):
    img_blur = cv.blur(img, (2, 2))
    img_cny = cv.Canny(img_blur, 50, 200, apertureSize=3)
    return img_cny


def thresh(img):
    thrsh, img_thresh = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    return img_thresh


# Probabilistic Hough Line Transform
def hough_lines_p(img):
    min_line_length = 12
    max_line_gap = 3
    lines = cv.HoughLinesP(
        img, cv.HOUGH_PROBABILISTIC, np.pi / 180, 0, None, min_line_length, max_line_gap
    )
    img_pre2 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # turn the picture fully black
    _, img_pre2 = cv.threshold(img_pre2, 255, 255, cv.THRESH_BINARY)
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv.line(img_pre2, (x1, y1), (x2, y2), (0, 128, 0), 1, cv.LINE_AA)
            # pts = np.array([[x1, y1], [x2, y2]], np.int32)
            # cv.polylines(img_pre2, [pts], True, (0, 255, 0))
    return lines


# detects endpoints and draws circles around them
def detect_end_points(img, cnts):
    """
    Detects endpoints and colors them grey. Only saves contours that have exactly two endpoints
    :param img:
    :param cnts:
    :return: endpnts: list with all contours and their corressponding two endpoints
    img_circle: the endpoint picture
    """
    img_circle = np.copy(img)
    img_circle = cv.cvtColor(img_circle, cv.COLOR_GRAY2BGR)
    endpnts = []
    # cnts = [contour.reshape((1, -1)) for contour in cnts]
    for cnt in cnts:
        cnt_endpnts = []
        for p in cnt:
            # p = p.reshape((1, -1))
            x = p[0][0]
            y = p[0][1]

            n = 0
            img_height = img.shape[0]
            img_width = img.shape[1]
            if x == img_width:
                print("hier!")
            if x != 0 and x != img_width - 1 and y != 0 and y != img_height - 1:
                n += img[y - 1, x]
                n += img[y - 1, x - 1]
                n += img[y - 1, x + 1]
                n += img[y, x - 1]
                n += img[y, x + 1]
                n += img[y + 1, x]
                n += img[y + 1, x - 1]
                n += img[y + 1, x + 1]
                n /= 255
                if n == 1:
                    cnt_endpnts.append(p)
                if n == 0:
                    cnt_endpnts.append(p)
        # filter out every finding of more than 2 or less than 2 endpoints per contour
        if len(cnt_endpnts) == 2:
            endpnts.append(cnt_endpnts)
            for p in cnt_endpnts:
                img_circle = cv.circle(img_circle, (p[0][0], p[0][1]), 0, (0, 0, 255))
                # cv.imshow("circles", img)
                # cv.waitKey(0)
        else:
            endpnts.append("")
    return endpnts, img_circle
    # cv.imshow('end_points', img)


# used to fill canny lines. after they get thinned, they might have a slight (1 pixel) offset to the original line
def dilate(img):
    img_tmp = np.copy(img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img_tmp = cv.dilate(img_tmp, kernel, iterations=1)
    return img_tmp


def erode(img):
    img_tmp = np.copy(img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img_tmp = cv.erode(img_tmp, kernel, iterations=1)
    return img_tmp


# remove duplicates from list
def remove_dupes(cnts):
    cnts_no_dupes = []
    for cnt in cnts:
        cnt = np.unique(cnt, axis=0)
        cnts_no_dupes.append(cnt)
    return cnts_no_dupes


# checks if contours are connected with length of distance between points
def check_connect(img, endpnts):
    connected_cntrs = []
    img_tmp = np.copy(img)
    # TO-DO: check if point was already connected
    checked_pnts = []
    for count, checkcnts in enumerate(endpnts):
        connected_pnts = []
        # hier checken
        final_dist = 0
        final_point = [0, 0]
        for count1, checkpnt in enumerate(checkcnts):
            x = checkpnt[0][0]
            y = checkpnt[0][1]
            dist = 20
            dist_pnt = []
            dist_count = None
            for count2, cnts in enumerate(endpnts):
                if cnts:
                    if (
                        checkcnts[0][0][0] != cnts[0][0][0]
                        or checkcnts[0][0][1] != cnts[0][0][1]
                    ):
                        for pnt in cnts:
                            x2 = pnt[0][0]
                            y2 = pnt[0][1]
                            # if x != x2 or y != y2:
                            dist2 = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
                            # if x2 == final_point[0] and y2 == final_point[1]:
                            #     if dist2 < final_dist:
                            if dist2 < dist:
                                dist = dist2
                                dist_pnt = [x2, y2]
                                dist_count = count2

            # if a matching point was found, delete point that was checked
            checked_pnts.append([x, y])
            if dist_pnt:
                # delete second point entry so it doesnt get connected again
                # del(endpnts[dist_count])
                cv.line(img_tmp, (x, y), (dist_pnt[0], dist_pnt[1]), 255, 1)
                # save the line that connects the points
                connected_pnts.append([[x, y], dist_pnt])
                checked_pnts.append(dist_pnt)
            # if no matching point was found
            else:
                connected_pnts.append([[x, y]])
                dist_pnt = [x, y]
            # if final_check == 0:
            #     final_check = dist
        connected_cntrs.append(connected_pnts)
    # cv.imshow("connected_lines", img)
    # cv.waitKey(0)
    return img_tmp, connected_cntrs


# help check if contours are connected by slightly stretching the lines
def increase_line(img, endpnts):
    img_tmp = np.copy(img)
    for cnts in endpnts:
        if len(cnts) == 2:
            x1 = cnts[0][0][0]
            y1 = cnts[0][0][1]
            x2 = cnts[1][0][0]
            y2 = cnts[1][0][1]
            vx = x1 - x2
            vy = y1 - y2
            div = 6
            while vx > 10 or vy > 10 or vx < -10 or vy < -10:
                vx //= div
                vy //= div
            cv.line(img_tmp, (x1, y1), (x1 + vx, y1 + vy), 255, 1)
            cv.line(img_tmp, (x2, y2), (x2 - vx, y2 - vy), 255, 1)
    return img_tmp


# finds contours, draws them, then saves them in an array and finally removes the duplicate points, returning a
# sorted array
def create_contours(img):
    cnts, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # draw the found contours
    img_clr = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # turn the picture fully black
    _, img_clr = cv.threshold(img_clr, 255, 255, cv.THRESH_BINARY)
    # draw the contours
    cv.drawContours(img_clr, cnts, -1, (0, 0, 255), 1)
    # get arrays of contour points
    cnts = get_cnts_array(cnts)
    # remove duplicates from contour list
    cnts = remove_dupes(cnts)
    return cnts, img_clr


# line segment detector
def get_lines_with_lsd(cnts, img, end_points):
    # Give contours and image
    lsd = cv.createLineSegmentDetector()
    all_lines = []
    all_hor_lines = []
    all_ver_lines = []
    img_lsd_full = np.zeros_like(img)
    img_hor_full = np.zeros_like(img)
    img_ver_full = np.zeros_like(img)
    for count, cnt in enumerate(cnts):
        if len(end_points[count]) > 3:
            del cnts[count]
            hor_lines = []
            ver_lines = []
            mask = np.zeros_like(img)
            cv.drawContours(mask, cnt, -1, (255, 0, 0), 2)
            # cv.imwrite(os.path.join(output_folder, f'a{count}_contour.png'), mask)
            # Detect with LSD
            lines, width, prec, nfa = lsd.detect(mask)
            img_lsd_part = np.zeros_like(mask)
            img_hor_lines = np.zeros_like(mask)
            img_ver_lines = np.zeros_like(mask)
            if lines is None:
                print(f"No lines found at contour {count}")
                continue
            for line in lines:
                x1, y1, x2, y2 = line[0].astype(np.int32)
                slope, length, is_hor = get_single_slope(line)
                cv.line(img_lsd_part, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv.line(img_lsd_full, (x1, y1), (x2, y2), (255, 255, 255), 1)
                # increase line length
                # vx = x2 - x1
                # vy = y2 - y1
                # x1 -= vx // 5
                # x2 += vx // 5
                # y1 -= vy // 5
                # y2 += vy // 5
                if is_hor:
                    hor_lines.append([[x1, y1], [x2, y2]])
                    cv.line(img_hor_lines, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv.line(img_hor_full, (x1, y1), (x2, y2), (255, 255, 255), 1)
                else:
                    ver_lines.append([[x1, y1], [x2, y2]])
                    cv.line(img_ver_lines, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv.line(img_ver_full, (x1, y1), (x2, y2), (255, 255, 255), 1)
            # cv.imwrite(os.path.join(output_folder, f'b{count}lsd_lines.png'), img_lsd_part)
            # cv.imwrite(os.path.join(output_folder, f'c{count}lsd_test1.png'), img_hor_lines)
            # cv.imwrite(os.path.join(output_folder, f'd{count}lsd_test2.png'), img_ver_lines)
            all_lines.append(lines)
            all_hor_lines.append(hor_lines)
            all_ver_lines.append(ver_lines)
    return (
        all_lines,
        img_lsd_full,
        all_hor_lines,
        all_ver_lines,
        img_hor_full,
        img_ver_full,
        cnts,
    )


def get_slope(lines):
    slps = []
    lngths = []
    for l in lines:
        # check if endpoints have been detected at all TO-DO: error with single endpoint
        if l:
            # x1, y1, x2, y2 = l[0][0].astype(np.int32)
            x1, y1, x2, y2 = l[0][0][0], l[0][0][1], l[1][0][0], l[1][0][1]
            sy = y2 - y1
            sx = x2 - x1
            if sx == 0:
                slope = 0
            else:
                slope = sy // sx
            length = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            slps.append(slope)
            lngths.append(length)
    return slps, lngths


def get_single_slope(line):
    x1, y1, x2, y2 = line[0].astype(np.int32)
    sy = y2 - y1
    sx = x2 - x1
    # horizontale linie, if sy > sx -> vertikale linie
    if abs(sx) > abs(sy):
        is_hor = True
    else:
        is_hor = False

    if sx == 0:
        slope = 0
    else:
        slope = sy // sx
    length = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    return slope, length, is_hor


def add_dummy_points(endpnts):
    dummy1 = [[[50, 123]]]
    dummy2 = [[[190, 280]]]
    dummy3 = [[[50, 133]]]
    endpnts.append(dummy1)
    endpnts.append(dummy2)
    endpnts.append(dummy3)
    return endpnts


# find partner contours for each contour and connect them
def check_cntr_pairs(connected_cntrs, endpnts):
    cntr_pairs = []
    for cntr_count, cntrs in enumerate(connected_cntrs):
        pairs = []
        for line in cntrs:
            if len(line) > 1:
                # x1 = line[0][0]
                # y1 = line[0][1]
                x = line[1][0]
                y = line[1][1]
                for end_cntr_count, end_cntr in enumerate(endpnts):
                    for point in end_cntr:
                        x_point = point[0][0]
                        y_point = point[0][1]
                        if x == x_point and y == y_point:
                            # both contours belong together
                            pairs.append(cntr_count)
                            pairs.append(end_cntr_count)
        cntr_pairs.append(pairs)
    return cntr_pairs


# joins the contour pairs into a single array
def join_cntr_pairs(cntr_pairs, num):
    pairs = []
    pair1 = cntr_pairs[num][0]
    pair2 = cntr_pairs[num][1]
    pairs.append(pair1)
    pairs.append(pair2)
    while cntr_pairs[pair2][1] not in pairs:
        pair2 = cntr_pairs[pair2][1]
        pairs.append(pair2)
    # pair1 is where we need to continue next
    return pairs, pair2


# joins all contour pairs into a single array
def join_all_cntr_pairs(cntr_pairs):
    num = 0
    all_pairs = []
    while num < len(cntr_pairs):
        pairs, num = join_cntr_pairs(cntr_pairs, num)
        all_pairs.append(pairs)
        num = num + 1
    return all_pairs


# finally check if the paired up contours form a dotted line or not
def check_dotted_line(cntr_pairs, cntr_lengths):
    is_pair_dotted = []
    for pairs in cntr_pairs:
        pair_lengths = []
        is_dotted = False
        for cntr in pairs:
            pair_lengths.append(cntr_lengths[cntr])
        pair_lengths = np.array(pair_lengths)
        lengths_mean = pair_lengths.mean()
        for l in pair_lengths:
            if lengths_mean - 5 < l < lengths_mean + 5:
                is_dotted = True
            else:
                is_dotted = False
            if len(pairs) < 4:
                is_dotted = False
        is_pair_dotted.append(is_dotted)
    return is_pair_dotted


def remove_dotted_from_endpnts(joined, isdotted, endpnts):
    endpnts_no_dotted = []
    for count, v in enumerate(isdotted):
        if not v:
            for cntrs in joined[count]:
                endpnts_no_dotted.append(endpnts[cntrs])
    return endpnts_no_dotted


#
# returns result dictionary for found object to object connections
def check_connect_matrix(img, endpnts, flat_endpnts, cntrs, objects):
    """
    uses new algorithm to increase endpoints and check for other endpoints in range
    check the contours with all points for the line, to determine the direction to increase the line with.
    then check for a endpoint in range to connect to
    :param img:
    :param endpnts:
    :param flat_endpnts:
    :param cntrs:
    :param objects:
    :return: result_dict: dictionary with found direct object to object connections
    :return: leftover_dict: dictionary with found endpoint to endpoint or endpoint to object connections
    """
    mappings = []
    img = np.copy(img)
    nodes_list = []
    edges_list = []
    endpnts_list = []
    endpnts_pair_list = []
    fp_l = []
    sp_l = []
    fc_l = []
    sc_l = []
    ob_l = []
    matrix_size = 5
    img_visualize = np.copy(img)
    img_visualize = cv.cvtColor(img_visualize, cv.COLOR_GRAY2BGR)
    line_colors = [
        np.random.randint(0, 256, size=3, dtype=np.uint8) for _ in range(len(endpnts))
    ]
    for count, pnts in enumerate(endpnts):
        first_found_obj = ""
        first_found_pnt = ""
        first_found_timeout = False
        if len(pnts) == 2:
            line_color = line_colors[count]
            for pnt in pnts:
                check_object = False
                center_point = (pnt[0][0], pnt[0][1])
                first_point, second_point = (
                    (pnts[0][0][0], pnts[0][0][1]),
                    (pnts[1][0][0], pnts[1][0][1]),
                )
                start_line = [first_point, second_point]
                (
                    img,
                    found_pnt,
                    found_obj,
                    connect_pnt,
                    obj_name,
                    mapper_a,
                ) = check_matrix(
                    img,
                    matrix_size,
                    center_point,
                    start_line,
                    flat_endpnts,
                    cntrs,
                    objects,
                    check_object,
                )
                for key, value in mapper_a.items():
                    mappings.append((key, value))

                next_pnts = get_contour_neighbour(center_point, cntrs, count)
                # set the timeout count
                initial_timeout = 30
                timeout = initial_timeout
                iterator = 0
                while timeout > 0:
                    if timeout < initial_timeout / 2 and not check_object:
                        iterator = 0
                        check_object = True
                        center_point = (pnt[0][0], pnt[0][1])
                        (
                            img,
                            found_pnt,
                            found_obj,
                            connect_pnt,
                            obj_name,
                            mapper_b,
                        ) = check_matrix(
                            img,
                            matrix_size,
                            center_point,
                            start_line,
                            flat_endpnts,
                            cntrs,
                            objects,
                            check_object,
                        )
                        for key, value in mapper_b.items():
                            mappings.append((key, value))

                    if found_pnt:
                        img_visualize = cv.circle(
                            img_visualize,
                            center_point,
                            0,
                            tuple(line_color.tolist()),
                            3,
                        )
                        if found_obj:
                            # idea: if Cardinality is found, proceed with line increase, until an endpoint in cardinal-
                            # ity was hit // check if both endpnts of cntr detect the SAME cntr+id?
                            # if obj_name[0] == 'C':
                            nodes_list.append(obj_name)
                            # if obj_name[0] == 'C':
                            img_visualize = cv.putText(
                                img_visualize,
                                obj_name,
                                center_point,
                                cv.FONT_HERSHEY_SIMPLEX,
                                1,
                                tuple(line_color.tolist()),
                                1,
                                2,
                            )
                            if first_found_obj == "":
                                first_found_obj = [obj_name, connect_pnt]
                            # endp1 found obj, endp2 found obj
                            else:
                                # checks that the line doesnt find connecting between same object
                                if first_found_obj[0] != obj_name:
                                    edges_list.append([first_found_obj[0], obj_name])
                                    endpnts_pair_list.append(
                                        [first_found_obj[1], connect_pnt]
                                    )
                                    endpnts_list.append(first_found_obj[1])
                                    endpnts_list.append(connect_pnt)
                            # endp1 found pnt or timeout, endp2 found obj
                            if first_found_pnt != "" or first_found_timeout:
                                if first_found_timeout:
                                    first_found_pnt = ""
                                (
                                    fp_l.append(first_point),
                                    sp_l.append(second_point),
                                    fc_l.append(first_found_pnt),
                                )
                                sc_l.append(""), ob_l.append(obj_name)
                        else:
                            if first_found_pnt == "":
                                first_found_pnt = connect_pnt
                            # endp1 found endp, obj or timeout, endp2 found endp
                            # endp1 found endp, endp2 found endp
                            else:
                                (
                                    fp_l.append(first_point),
                                    sp_l.append(second_point),
                                    fc_l.append(first_found_pnt),
                                )
                                sc_l.append(connect_pnt), ob_l.append("")
                            # endp 1 found obj or timeout, endp2 found endp
                            if first_found_obj != "" or first_found_timeout:
                                if first_found_timeout:
                                    obj = ""
                                else:
                                    obj = first_found_obj[0]
                                (
                                    fp_l.append(first_point),
                                    sp_l.append(second_point),
                                    fc_l.append(""),
                                )
                                sc_l.append(connect_pnt), ob_l.append(obj)

                        break
                    center_point = (
                        center_point[0] + next_pnts[iterator][0],
                        center_point[1] + next_pnts[iterator][1],
                    )
                    (
                        img,
                        found_pnt,
                        found_obj,
                        connect_pnt,
                        obj_name,
                        mapper_c,
                    ) = check_matrix(
                        img,
                        matrix_size,
                        center_point,
                        start_line,
                        flat_endpnts,
                        cntrs,
                        objects,
                        check_object,
                    )
                    for key, value in mapper_c.items():
                        mappings.append((key, value))

                    if iterator == len(next_pnts) - 1:
                        iterator = 0
                    else:
                        iterator += 1
                    timeout -= 1
                    # todo: timeout <= initial_timeout / 2 might not apply for weird numbers
                    if timeout == 0:
                        if first_found_pnt != "":
                            (
                                fp_l.append(first_point),
                                sp_l.append(second_point),
                                fc_l.append(first_found_pnt),
                            )
                            sc_l.append(""), ob_l.append("")
                        elif first_found_obj != "":
                            (
                                fp_l.append(first_point),
                                sp_l.append(second_point),
                                fc_l.append(""),
                            )
                            sc_l.append(""), ob_l.append(first_found_obj[0])
                        else:
                            first_found_timeout = True
    leftover_dict = {
        "first_point": fp_l,
        "second_point": sp_l,
        "first_connect": fc_l,
        "second_connect": sc_l,
        "object": ob_l,
    }
    # todo: filter out cardinality paired with anything / adjust dict for cardinality
    result_dict = {
        "nodes": list(set(nodes_list)),
        "edges": edges_list,
        "endpoints": endpnts_list,
        "endpoint_pairs": endpnts_pair_list,
    }

    return img, img_visualize, result_dict, leftover_dict, mappings


def check_matrix(
    img,
    matrix_size,
    center_point,
    start_line,
    flat_endpnts,
    cntrs,
    objects,
    check_object,
):
    """
    Check if the matrix around the given center_point finds a connecting point or object
    :param img:
    :param matrix_size:
    :param center_point:
    :param start_line:
    :param flat_endpnts:
    :param cntrs:
    :param objects:
    :param check_object:
    :return: img: the image with the added pixel on the currently inspected center_point
    found_point: a boolean that shows if a point was found
    found_object: a boolean that shows if an object was found
    connect_point: the coordinates of the found point, returns [0, 0] if none was found
    object_name: the name of the object, returns 'None' if none was found
    """
    mapper = {}
    found_point = False
    found_object = False
    connect_point = [0, 0]
    object_name = "None"
    card_found = "None"
    # checks if point is at edge of picture
    if (
        center_point[0] > img.shape[1] - 4
        or center_point[0] < 4
        or center_point[1] > img.shape[0] - 4
        or center_point[1] < 4
    ):
        return img, found_point, found_object, connect_point, object_name, mapper
    matrix = [[0] * matrix_size for _ in range(matrix_size)]
    start_x = center_point[0] - (matrix_size - 1) // 2
    start_y = center_point[1] - (matrix_size - 1) // 2
    for j in range(matrix_size):
        for k in range(matrix_size):
            if found_point:
                break
            matrix_x, matrix_y = start_x + j, start_y + k
            # save point in matrix list
            matrix[j][k] = (matrix_x, matrix_y)
            # check for connection to cut-out objects and then for endpoints
            # todo: once an endpnt got a partner, it shouldnt be able to find another
            # important to make sure it doesnt find the endpoints from the line it originated from
            if matrix[j][k] != start_line[0] and matrix[j][k] != start_line[1]:
                if check_object:
                    for count, obj in enumerate(objects):
                        if (
                            obj.segmentation[(matrix_y, matrix_x)]
                            and obj.classification != "Cardinality"
                        ):
                            # filter out
                            if obj.classification == "Unrecognizable":
                                break
                            found_point = True
                            found_object = True
                            connect_point = matrix[j][k]
                            object_name = f"{obj.classification[0]}{count}"
                            # print(f"found connection to {object.classification} with point {connect_point}!")
                            mapper[f"{obj.classification[0]}{count}"] = obj
                            break
                        if (
                            obj.segmentation[(matrix_y, matrix_x)]
                            and obj.classification == "Cardinality"
                        ):
                            card_found = f"{obj.classification[0]}{count}"
                            mapper[f"{obj.classification[0]}{count}"] = obj

                elif not found_point and matrix[j][k] in flat_endpnts:
                    # if card_found != 'None':
                    #     print("found a cardinality")
                    found_point = True
                    found_object = False
                    connect_point = matrix[j][k]
                    # print(f"found connection to another line with point {connect_point}!")
                    break
    img = cv.circle(img, center_point, 0, 255, 1)
    if not found_point:
        return img, found_point, found_object, connect_point, object_name, mapper
    else:
        # print(f"found a match: {connect_point}")
        return img, found_point, found_object, connect_point, object_name, mapper


def get_contour_neighbour(pnt, cntrs, cntrs_pos):
    """
    Searches contour array for neighbours of given pixel and returns array with chain of five neighbours in order to
    gauge the direction of the given line
    :param pnt:
    :param cntrs:
    :param cntrs_pos:
    :return: neighbours: gives the direction the endpoint needs to get increased by, to follow the found structure
    """
    neighbours = []
    index = 0
    go_inc = False
    for count, pnts in enumerate(cntrs[cntrs_pos]):
        if pnts[0][0] == pnt[0] and pnts[0][1] == pnt[1]:
            index = count
    check_index = index + 1
    if len(cntrs[cntrs_pos]) == check_index:
        check_index = 0
    try_inc_x, try_inc_y = (
        pnt[0] - cntrs[cntrs_pos][check_index][0][0],
        pnt[1] - cntrs[cntrs_pos][check_index][0][1],
    )
    # try_dec_x, try_dec_y = pnt[0] - cntrs[cntrs_pos][index-1][0][0], pnt[1] - cntrs[cntrs_pos][index-1][0][1]
    if 1 >= try_inc_x >= -1 and 1 >= try_inc_y >= -1:
        go_inc = True
    else:
        go_inc = False
    count = 5
    if len(cntrs[cntrs_pos]) < 10:
        count = (len(cntrs[cntrs_pos]) // 2) - 1
    while count >= 0:
        if go_inc:
            check_index = index + count
            check_next_index = check_index + 1
            # checks for inc
            if len(cntrs[cntrs_pos]) <= check_index:
                check_index = check_index - len(cntrs[cntrs_pos])
                check_next_index = check_index + 1
            if len(cntrs[cntrs_pos]) <= check_next_index:
                check_next_index = check_next_index - len(cntrs[cntrs_pos])
        else:
            check_index = index - count
            check_next_index = check_index - 1
            # checks for dec
            if check_index == -1:
                check_index = len(cntrs[cntrs_pos]) + check_index
            if check_index < -1:
                check_index = len(cntrs[cntrs_pos]) + check_index
                check_next_index = check_index + 1
            if check_next_index == 0:
                check_next_index = len(cntrs[cntrs_pos]) - 1
        dir = [
            cntrs[cntrs_pos][check_index][0][0]
            - cntrs[cntrs_pos][check_next_index][0][0],
            cntrs[cntrs_pos][check_index][0][1]
            - cntrs[cntrs_pos][check_next_index][0][1],
        ]
        neighbours.append(dir)
        count -= 1
    return neighbours


def combine_leftover_lines(leftover_dict):
    """
    Takes the leftover dictionary and combines the found lines that connect to each other into one structure with two
    endpoints
    :param leftover_dict:
    :return: leftover_dict: an updated leftover dictionary
    """
    fp_l = leftover_dict.setdefault("first_point")
    sp_l = leftover_dict.setdefault("second_point")
    fc_l = leftover_dict.setdefault("first_connect")
    sc_l = leftover_dict.setdefault("second_connect")
    ob_l = leftover_dict.setdefault("object")
    fp_comb_l, sp_comb_l, fc_comb_l, sc_comb_l, ob_comb_l = [], [], [], [], []
    pair_list = []
    index_list = []
    pair_list = search_detected_list(fc_l, fp_l, sp_l, pair_list)
    pair_list = search_detected_list(sc_l, fp_l, sp_l, pair_list)
    # get the index numbers for the contours that connect to each other
    for count1, (x, y) in enumerate(pair_list):
        # temp_list = []
        temp_tuple = ()
        index_l_match = False
        for count2, item in enumerate(pair_list):
            if x in item or y in item:
                # temp_list.append(item)
                temp_tuple = tuple(set(temp_tuple + item))
                # get rid of duplicates
                for count3, elem in enumerate(index_list):
                    if item[0] in elem or item[1] in elem:
                        temp_tuple = tuple(set(temp_tuple + index_list[count3]))
                        # get rid of duplicates
                        index_list[count3] = temp_tuple
                        index_l_match = True
                # pair_list[count2] = (-1, -1)
        # if temp_list:
        # index_list.append(temp_list)
        if not index_l_match:
            index_list.append(temp_tuple)
    # flatten the list and only save unique index numbers
    for elem in index_list:
        temp_l1, temp_l2, temp_l3, temp_l4, temp_l5 = [], [], [], [], []
        # if a line finds an object, save only the endpoint that found it
        for index in elem:
            (
                temp_l1.append(fp_l[index]),
                temp_l2.append(sp_l[index]),
                temp_l3.append(fc_l[index]),
            )
            temp_l4.append(sc_l[index]), temp_l5.append(ob_l[index])
        fp_comb_l.append(temp_l1), sp_comb_l.append(temp_l2), fc_comb_l.append(temp_l3)
        sc_comb_l.append(temp_l4), ob_comb_l.append(temp_l5)
    leftover_dict = {
        "first_point": fp_comb_l,
        "second_point": sp_comb_l,
        "first_connect": fc_comb_l,
        "second_connect": sc_comb_l,
        "object": ob_comb_l,
    }
    return leftover_dict


def search_detected_list(li, fp_l, sp_l, pair_list):
    """
    Used in combine_leftover_lines and checks if the connected point in li is contained in either fp_l or sp_l. Those
    that are contained, get added to pair_list
    :param li: either first connect or second connect
    :param fp_l:
    :param sp_l:
    :param pair_list:
    :return: pair_list: which contains all found matches between li and fp_l or sp_l
    """
    for count, pnt in enumerate(li):
        if pnt != "":
            fp_c = fp_l.count(pnt)
            sp_c = sp_l.count(pnt)
            if fp_c > 1 or sp_c > 1:
                print("error: too many finds")
            if fp_c == 1:
                index = fp_l.index(pnt)
                if index != count:
                    pair_list.append((count, index))
            if sp_c == 1:
                index = sp_l.index(pnt)
                # if index == count:
                #     print(f"found itself index: {count}")
                if index != count:
                    pair_list.append((count, index))
            if sp_c == 1 and fp_c == 1:
                print("error: found pair in both lists")
    return pair_list


def get_cardinality_endpnts(objects, img, end_points, contours):
    """
    Takes mask for cardinality and detects the endpoints for its line
    :param objects: the cut-out objects
    :param img: the full image without the objects cut out
    :param end_points: all the previously found endpoints
    :return: returns the two endpoints for the line
    """
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_masked = np.zeros_like(img_gray)

    # taken from segmentation/segmentation.py
    for elem in objects:
        if elem.classification == "Cardinality":
            mask = dilation(elem.segmentation, square(3))
            img_masked[mask] = img_gray[mask]
    _, img_thresh = cv.threshold(img_masked, 170, 255, cv.THRESH_BINARY)
    img_thinned = cv.ximgproc.thinning(img_thresh)
    img_dilate = dilate(img_thinned)
    img_erode = erode(img_dilate)
    img_thinned = cv.ximgproc.thinning(img_erode)
    card_contours = get_contours(img_thinned)
    endpnts, img_endpoints = detect_end_points(img_thinned, card_contours)
    for elem in endpnts:
        end_points.append(elem)
    all_cntrs = contours + card_contours
    non_thinned_card_cntrs = get_contours(img_masked)
    return end_points, all_cntrs, img_endpoints, non_thinned_card_cntrs


def get_result_from_leftover(combined_leftover_dict, img, result_dict, orig_img):
    """
    Go through the combined leftover dictionary and add the lines that found two objects to the result dictionary, while
    creating debug images
    :param combined_leftover_dict:
    :param img:
    :param result_dict:
    :return: img_leftover, img_leftover2, img_leftover3, img_leftover4:
    image with lines that found 1 to 4 objects respectively
    :return: result_dict: the result dictionary with the added lines that found two connecting objects
    """
    # draw the leftover endpoints, that found other endpoints or only 1 object
    img_leftover = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img_leftover2 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img_leftover3 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img_leftover4 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    line_colors = [
        np.random.randint(0, 256, size=3, dtype=np.uint8)
        for _ in range(len(combined_leftover_dict["first_point"]))
    ]
    for count1, pnt_l in enumerate(combined_leftover_dict["first_point"]):
        line_color = tuple(line_colors[count1].tolist())
        obj_list = []
        endpnts_list = []
        endpnts_pair = []
        img_loop = np.zeros_like(img_leftover)
        for count2, pnt in enumerate(pnt_l):
            obj = combined_leftover_dict["object"][count1][count2]
            pnt2 = combined_leftover_dict["second_point"][count1][count2]
            if obj != "":
                # only add endpoints to the list that actually found the object
                obj_list.append(obj)
                if (
                    combined_leftover_dict["first_connect"][count1][count2] == ""
                    and combined_leftover_dict["second_connect"][count1][count2] == ""
                ):
                    img_loop = cv.circle(img_loop, pnt, 0, line_color, 5)
                    endpnts_list.append(pnt2)
                    endpnts_pair.append(pnt2)
                elif combined_leftover_dict["first_connect"][count1][count2] == "":
                    img_loop = cv.circle(img_loop, pnt2, 0, line_color, 5)
                    endpnts_list.append(pnt)
                    endpnts_pair.append(pnt)
                elif combined_leftover_dict["second_connect"][count1][count2] == "":
                    img_loop = cv.circle(img_loop, pnt2, 0, line_color, 5)
                    endpnts_list.append(pnt2)
                    endpnts_pair.append(pnt2)
            else:
                img_loop = cv.circle(img_loop, pnt, 0, line_color, 5)
                img_loop = cv.circle(img_loop, pnt2, 0, line_color, 5)
                # endpnts_list.extend([pnt, pnt2])
                # endpnts_pair.append(pnt)
                # endpnts_pair.append(pnt2)
        # only draw the points, if it found exactly 2 objects
        # gets rid of duplicates
        obj_list = list(set(obj_list))
        # found 1 object, so its trash or did not get connected properly
        if len(obj_list) == 1:
            img_leftover = cv.bitwise_or(img_leftover, img_loop)
        # found 2 objects that are not the same; very good!
        elif len(obj_list) == 2 and obj_list[0] != obj_list[1]:
            img_leftover2 = cv.bitwise_or(img_leftover2, img_loop)
            for elem in obj_list:
                result_dict["nodes"].append(elem)
                result_dict["nodes"] = list(set(result_dict["nodes"]))
            result_dict["edges"].append(obj_list)
            for pnt in endpnts_list:
                result_dict["endpoints"].append(pnt)
            result_dict["endpoint_pairs"].append(endpnts_pair)
        # found 3 objects, should be good, one object probably got found twice
        elif len(obj_list) == 3:
            img_leftover3 = cv.bitwise_or(img_leftover3, img_loop)
        # trash or did not get connected properly
        else:
            img_leftover4 = cv.bitwise_or(img_leftover4, img_loop)

    # draw the filtered endpoints that have found 2 connecting objects
    line_colors = []
    n_colors = len(result_dict["endpoint_pairs"])
    colors = [cm.rainbow(x) for x in np.linspace(0, 1, n_colors)]

    for color in colors:
        # Convert RGBA to BGR
        bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        line_colors.append(bgr_color)

    # line_colors = [
    #     np.random.randint(0, 256, size=3, dtype=np.uint8)
    #     for _ in range(len(result_dict["endpoint_pairs"]))
    # ]
    img_result = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for count, data in enumerate(
        zip(result_dict["endpoint_pairs"], result_dict["edges"])
    ):
        pair, label = data
        line_color = line_colors[count]
        dist = int(distance.euclidean(pair[0], pair[1]))

        for idx, pnt in enumerate(pair):
            # img_result = cv.circle(img_result, pnt, 0, tuple(line_color.tolist()), 5)
            img_result = cv.circle(orig_img, pnt, 4, line_color, -1)

            if idx == 0:
                # img_result = cv.putText(
                #     img_result,
                #     label[idx],
                #     pnt,
                #     cv.FONT_HERSHEY_SIMPLEX,
                #     0.4,
                #     line_color,
                #     1,
                #     2,
                # )
                img_result = cv.putText(
                    img_result,
                    str(dist),
                    (pnt[0] + 10, pnt[1]),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    line_color,
                    2,
                    2,
                )
            # check connect matrix
    return (
        img_leftover,
        img_leftover2,
        img_leftover3,
        img_leftover4,
        img_result,
        result_dict,
    )
