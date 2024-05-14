from itertools import chain

import cv2 as cv
import line_detection.utils as utils
import numpy as np


def detect_lines(img, objects, img_noncut):
    # get one pixel thin lines
    img_thinned = cv.ximgproc.thinning(img)
    # find and draw contours
    contours = utils.get_contours(img_thinned)
    contours = [contour for contour in contours if len(contour) > 0]

    # debug imgs
    intersec_img = cv.cvtColor(img_thinned, cv.COLOR_GRAY2BGR)
    cont_img = np.zeros_like(intersec_img)
    split_cont_img = np.zeros_like(intersec_img)
    split_cont_with_end_img = np.zeros_like(intersec_img)

    # original contour image
    for i, contour in enumerate(contours):
        color = tuple(
            np.random.randint(0, 255, 3).tolist()
        )  # generate random color for each contour
        cv.drawContours(cont_img, [contour], -1, color, thickness=0)

    # find intersections
    centroids = []
    all_crossings_with_cont = utils.find_crossings(contours, img_thinned, 3)
    all_crossings = [
        entry["point"] for entry in all_crossings_with_cont if entry is not None
    ]

    if len(all_crossings) > 0:
        centroids = utils.group_and_calculate_centroids(
            all_crossings, img_thinned, all_crossings_with_cont, distance_threshold=10
        )
        if len(centroids) > 0:
            contours = utils.split_all_contours(contours, centroids, img_thinned)

    contours = tuple([contour for contour in contours if contour.any()])

    # find end points for contours formed by endpoints
    end_points, img_endpoints = utils.detect_end_points(img_thinned, contours)

    split_cont_img, split_cont_with_end_img = get_inter_images(
        contours,
        end_points,
        centroids,
        split_cont_img,
        split_cont_with_end_img,
        intersec_img,
    )

    # get endpoints for cardinality and add them to the other endpoints
    # and get their contours and add them to the other contours
    (
        end_points,
        contours,
        img_card_endpnts,
        contours_non_thinned_card,
    ) = utils.get_cardinality_endpnts(objects, img_noncut, end_points, contours)
    # get contours of non thinned image for labeling purposes
    contours_non_thinned = utils.get_contours(img)
    contours_all_non_thinned = contours_non_thinned + contours_non_thinned_card
    # flatten endpoint list for easier endpoint finding in the list
    flattened_endpoints = list(chain(*end_points))
    flattened_endpoints = {
        tuple(point) for endpnt in flattened_endpoints for point in endpnt
    }
    # slopes, lengths = get_slope(end_points)
    # check for connections between endpoints and objects
    (
        img_increase,
        img_visualize,
        result_dict,
        leftover_dict,
        mappings,
    ) = utils.check_connect_matrix(
        img_thinned, end_points, flattened_endpoints, contours, objects
    )
    # check the leftover dictionary and combine the pieces accordingly
    combined_leftover_dict = utils.combine_leftover_lines(leftover_dict)
    img_endpoints_full = cv.bitwise_or(img_endpoints, img_card_endpnts)
    # add the combined leftover lines that found two objects to the result dictionary
    (
        img_leftover,
        img_leftover2,
        img_leftover3,
        img_leftover4,
        img_result,
        result_dict,
    ) = utils.get_result_from_leftover(
        combined_leftover_dict, img, result_dict, img_noncut
    )
    # # uncomment for debug images
    draw_debug_imgs(
        img_card_endpnts,
        img_endpoints,
        img_endpoints_full,
        img_increase,
        img_leftover,
        img_leftover2,
        img_leftover3,
        img_leftover4,
        img_noncut,
        img_result,
        img_thinned,
        img_visualize,
        split_cont_img,
        intersec_img,
    )

    center_point = (445, 872)
    size = 200
    crop_intersection(
        center_point,
        size,
        img,
        img_thinned,
        cont_img,
        split_cont_img,
        split_cont_with_end_img,
    )

    return result_dict, contours_all_non_thinned, img_result, all_crossings, mappings


def get_inter_images(
    contours,
    end_points,
    centroids,
    split_cont_img,
    split_cont_with_end_img,
    intersec_img,
):
    # draw Split contour image with endpoints
    for i, (contour, endpoints) in enumerate(zip(contours, end_points)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv.drawContours(split_cont_img, [contour], -1, color, thickness=0)
        cv.drawContours(split_cont_with_end_img, [contour], -1, color, thickness=0)
        for endpoint in endpoints:
            x, y = endpoint[0]
            cv.circle(
                split_cont_with_end_img, (x, y), radius=2, color=color, thickness=-1
            )
    # with Intersection
    for point in centroids:
        cv.circle(
            split_cont_with_end_img, point, radius=3, color=(0, 0, 255), thickness=-1
        )
        cv.circle(split_cont_img, point, radius=3, color=(0, 0, 255), thickness=-1)
    cv.imwrite("output/inter_new_lines.png", split_cont_img)
    cv.imwrite("output/intersections.png", intersec_img)
    return split_cont_img, split_cont_with_end_img


def crop_intersection(center_point, size, *images, name=""):
    x, y = center_point
    for idx, img in enumerate(images):
        cropped_image = img[
            int(x - size / 2) : int(x + size / 2), int(y - size / 2) : int(y + size / 2)
        ]
        # cv.imwrite(f"cropped_img_{name}{idx+1}.png", cropped_image)


def draw_debug_imgs(
    img_card_endpnts,
    img_endpoints,
    img_endpoints_full,
    img_increase,
    img_leftover,
    img_leftover2,
    img_leftover3,
    img_leftover4,
    img_noncut,
    img_result,
    img_thinned,
    img_visualize,
    split_cont_img,
    intersec_img,
):
    cv.imwrite("output/endpoints.png", img_endpoints)
    cv.imwrite("output/endpoints_full.png", img_endpoints_full)
    cv.imwrite("output/endpoints_card.png", img_card_endpnts)
    cv.imwrite("output/visualize.png", img_visualize)
    cv.imwrite("output/increase.png", img_increase)
    cv.imwrite("output/thinned.png", img_thinned)
    cv.imwrite("output/noncut.png", img_noncut)
    cv.imwrite("output/leftover.png", img_leftover)
    cv.imwrite("output/leftover2.png", img_leftover2)
    cv.imwrite("output/leftover3.png", img_leftover3)
    cv.imwrite("output/leftover4.png", img_leftover4)
    cv.imwrite("output/result.png", img_result)
    cv.imwrite("output/inter_new_lines.png", split_cont_img)
    cv.imwrite("output/intersections.png", intersec_img)
