import json

import cv2 as cv
import numpy as np

# todo check endpoints list, only keep 4 outer points of matrix

# todo loop through all images
# todo upload and export all images, grab image url/id from json
# todo insert into jsons here


def is_inside_contours(point, contours):
    x, y = point
    x_float = float(x)
    y_float = float(y)
    for contour in contours:
        if cv.pointPolygonTest(contour, (x_float, y_float), False) >= 0:
            return True, contour
    return False, contour


def normalize_coordinates(point, img_shape):
    height, width = img_shape[:2]
    x_norm = (point[0] / width) * 100
    y_norm = (point[1] / height) * 100
    return x_norm, y_norm


def save_to_json(endpoints, valid_contours, image_name, count):
    with open("../json/input.json", "r") as json_file:
        data = json.load(json_file)
    pos = 0
    for index, element in enumerate(data):
        name = element["data"]["image"]
        if image_name in name:
            pos = index
    prepare = {"result": []}
    data[pos]["predictions"].append(prepare)

    for index, valid_contour in enumerate(valid_contours, start=1):
        formatted_contour = [
            [x, y] for x, y in zip(valid_contour[::2], valid_contour[1::2])
        ]
        # todo create new dict for image url entry
        prediction_entry = {
            "id": f"prediction_{index}",
            "type": "polygonlabels",
            "value": {
                "closed": True,
                "points": formatted_contour,
                "polygonlabels": ["Line"],
            },
            "origin": "manual",
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 1024,
            "original_height": 475,
        }
        # find point where image_name is in string of data path
        data[pos]["predictions"][0]["result"].append(prediction_entry)

    for index, endpoint_frame in enumerate(endpoints, start=len(valid_contours) + 1):
        endpoint_points = [
            [x, y] for x, y in zip(endpoint_frame[::2], endpoint_frame[1::2])
        ]
        # todo repeat new entry step from above
        prediction_entry = {
            "id": f"prediction_{index}",
            "type": "polygonlabels",
            "value": {
                "closed": True,
                "points": endpoint_points,
                "polygonlabels": ["Endpoint"],
            },
            "origin": "manual",
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 1024,
            "original_height": 475,
        }

        data[pos]["predictions"][0]["result"].append(prediction_entry)

    with open("../json/input.json", "w") as json_file:
        json.dump(data, json_file, indent=2)


def get_lines(endpoints, contours, img):
    width, height = img.shape[:2]
    blank_image = np.zeros((height, width, 3), np.uint8)
    valid_contours = []
    matrix_size = 5
    all_valid_points = []
    for endpoint in endpoints:
        valid_points = []
        x, y = endpoint
        cv.circle(blank_image, (x, y), 1, (0, 0, 255), -1)
        for dy in range(-matrix_size // 2, (matrix_size // 2) + 1):
            for dx in range(-matrix_size // 2, (matrix_size // 2) + 1):
                point = (x + dx, y + dy)
                is_inside_contour, contour = is_inside_contours(point, contours)
                if is_inside_contour:
                    contours_strings = [
                        np.array2string(valid_contour)
                        for valid_contour in valid_contours
                    ]
                    contour_to_check_string = np.array2string(contour)
                    valid_points.append(point)
                    if contour_to_check_string not in contours_strings:
                        valid_contours.append(contour)
        all_valid_points.append(valid_points)

    cv.drawContours(blank_image, valid_contours, -1, (0, 255, 0), 1)
    # cv.imwrite("img2.png", blank_image)

    # todo change endpoints to optimal values from all_valid_points
    endpoints_normalized = []
    for endpoint in endpoints:
        x, y = endpoint
        endpoint_normalized = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x_matrix = x + dx
                y_matrix = y + dy
                normalized_x_matrix, normalized_y_matrix = normalize_coordinates(
                    (x_matrix, y_matrix), img.shape
                )
                endpoint_normalized.extend(
                    [round(normalized_x_matrix, 3), round(normalized_y_matrix, 3)]
                )
        endpoints_normalized.append(endpoint_normalized)

    contours_normalized = []
    for contour_idx, contour in enumerate(valid_contours):
        contour_normalized = []
        for point in contour:
            normalized_x, normalized_y = normalize_coordinates(point[0], img.shape)
            contour_normalized.extend([round(normalized_x, 3), round(normalized_y, 3)])
        contours_normalized.append(contour_normalized)

    return endpoints_normalized, contours_normalized

    # YOLO format, can't upload to LabelStudio
    # with open('labels.txt', 'w') as file:
    #     for label in endpoint_labels + contour_labels:
    #         label_str = ' '.join(map(str, label))
    #         file.write(label_str + '\n')
