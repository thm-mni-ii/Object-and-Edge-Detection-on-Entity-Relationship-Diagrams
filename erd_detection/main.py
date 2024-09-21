import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from line_detection.line import detect_lines
from object_detection.detection import cut_boxes, get_bboxes
from preprocessing import yolo_preprocess, trocr_preprocessor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segmentation.segmentation import (
    create_mask_information,
    get_masks,
    get_residual_elements,
    postprocess_masks,
    remove_erd_elements,
    tailor_image,
)
from torchvision.ops import nms
from ultralytics import YOLO
from text_detection.inference import OCRModel

# YOLO
yolo_weights = Path("./weights/yolo.pt")
assert yolo_weights.exists()
model = YOLO(yolo_weights)

# SAM
model_type = "vit_h"
device = "cuda"
sam_weights = Path("./weights/sam_vit_h_4b8939.pth")

sam = sam_model_registry[model_type](checkpoint=sam_weights)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=78,
)

# TrOCR
trocr_weights = Path("./weights/b-low-lr.ckpt")

trocr = OCRModel.load_from_checkpoint(
    checkpoint_path=trocr_weights,
)
trocr = trocr.to(device)


Path("output").mkdir(exist_ok=True)

dataset = Path("./data")
assert dataset.exists()

for file in dataset.iterdir():
    if file.is_file():
        image_orig = cv.imread(str(file))
        image_prep = yolo_preprocess(image_orig)

        ### object detection
        detection_info, id2label = get_bboxes([image_prep], model)
        results = cut_boxes([image_prep], detection_info, [2, 1])

        ### additional non-maximum suppression technique
        IOU = 0.3
        results_per_image = [
            (
                torch.tensor(np.array([t[0] for t in detection])),
                torch.tensor(np.array([t[1] for t in detection])),
                torch.tensor(np.array([t[2] for t in detection])),
            )
            for detection in detection_info
        ]

        removals_per_image = []
        for result in results_per_image:
            all_idx = set(range(result[0].shape[0]))
            sel_idx = set(nms(result[0], result[1], IOU).numpy())

            rem_idx = torch.tensor(list((all_idx.difference(sel_idx))), dtype=int)

            removals = torch.index_select(result[0], 0, rem_idx)
            removals_per_image.append(removals.to(int).tolist())

        ### segmentation
        masks = get_masks(results[0], mask_generator)

        masks, classifications_a = postprocess_masks(image_prep, masks, detection_info)
        resids, classifications_b = get_residual_elements(
            masks, detection_info, image_prep[:, :, 0].shape
        )
        elements = create_mask_information(
            np.append(masks, resids),
            np.append(classifications_a, classifications_b),
            id2label,
        )

        removal_list = []
        for removal_obj in removals_per_image[0]:
            obj_dst = {}

            for detected_obj in elements:
                removal_obj_arr = np.array(removal_obj)
                detected_obj_arr = np.array(detected_obj.bbox)

                diff = np.sum(
                    np.abs(np.array(removal_obj) - np.array(detected_obj.bbox))
                )
                obj_dst[diff] = detected_obj

            min_key = min(obj_dst.keys())
            removal_list.extend([obj_dst[min_key]])

        elements_nms = list(set(elements) - set(removal_list))

        # algorithmic line detection
        for count, elem in enumerate(elements_nms):
            name = f"{elem.classification[0]}{count}"
            point = (int(elem.bbox[0]), int(elem.bbox[1]))
            temp_img = np.zeros_like(results[0])
            temp_img = cv.putText(
                temp_img, name, point, cv.FONT_HERSHEY_SIMPLEX, 1, 255, 1, 2
            )
            results[0] = cv.bitwise_or(temp_img, results[0])

        line_detection_image = remove_erd_elements(image_prep, elements)
        line_detection_image = tailor_image(line_detection_image, elements)
        result_dict, contours_all_non_thinned, result_img, _, mappings = detect_lines(
            line_detection_image, elements, image_prep
        )

        mappings = list(set(mappings))

        # Add text inference
        preprocessor = trocr_preprocessor(image_orig, image_prep)
        for mapping in mappings:
            img = preprocessor(mapping[1])
            if mapping[1].classification != "Clearly Unrecognizable":
                text = trocr.detect_text(img)
                mapping[1].text = text

        to_serialize = [mapping[1].to_dict(mapping[0]) for mapping in mappings]

        data = {"Nodes": to_serialize, "Edges": result_dict["edges"]}

        with open(f"output/{file.stem}.json", "w") as file:
            json.dump(data, file)
