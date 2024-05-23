import cv2 as cv
import numpy as np
import torch
from shapely.geometry import box
from skimage.morphology import dilation, square
from torchvision.ops import box_convert


class DetectedMask:
    def __init__(self, mask_info, classification, id2label):
        self.segmentation = mask_info["segmentation"]
        self.bbox = mask_info["bbox"]
        self.classification = id2label[int(classification)]
        self.text = ""

    def __str__(self):
        return f"{self.classification} detected inside {self.bbox}"

    def to_dict(self, identifier):
        data = {
            "id": identifier,
            "classification": self.classification,
            "bbox": [str(x) for x in list(self.bbox)],
            "text": self.text,
        }
        return data


def get_masks(image, mask_generator):
    return mask_generator.generate(image)


def postprocess_masks(image, masks, bboxes):
    """
    Keep masks which are only as big as a quarter of the image
    and match 80% of a detected bbox
    """

    masks = [
        mask for mask in masks if mask["area"] <= (image.shape[0] * image.shape[1]) / 4
    ]

    classifications = []
    cleaned_masks = []
    for mask in masks:
        bbox_xyxy = box_convert(
            torch.tensor(mask["bbox"]), "xywh", "xyxy"
        ).numpy()  # xywh -> xyxy
        bbox_sam = box(*bbox_xyxy)

        for bbox_yolo in bboxes[0]:
            bbox_yolo, _, classification = bbox_yolo

            bbox_yolo = box(*bbox_yolo)
            intersection = bbox_sam.intersection(bbox_yolo).area
            union = bbox_sam.area + bbox_yolo.area - intersection
            iou = intersection / union if union != 0 else 0

            if iou > 0.80:
                mask["bbox"] = bbox_xyxy
                cleaned_masks.append(mask)
                classifications.append(classification)
                break

    return cleaned_masks, classifications


def get_residual_elements(masks, bboxes, shape):
    """
    Find masks which could not be detectet by sam
    or which should be added without segmentation
    """

    # Elements found and kept by mask postprocessing
    masks_bbox = [box(*element["bbox"]) for element in masks]

    # All elements found by yolo which we assume all to be correct
    bboxes = [
        (box(*bbox_yolo), int(classification))
        for bbox_yolo, _, classification in bboxes[0]
    ]
    residual_bboxes = bboxes.copy()

    # Find elements from the bbox detector (yolo), which does not share 80%
    # of area with one of the segmentation bbox.
    # Those elements are considered not detected.
    # The segmentation is equal to the full bbox
    for element in bboxes:
        bbox_yolo, classification = element
        for bbox_sam in masks_bbox:
            intersection = bbox_sam.intersection(bbox_yolo).area
            union = bbox_sam.area + bbox_yolo.area - intersection
            iou = intersection / union if union != 0 else 0

            if iou > 0.80:
                residual_bboxes.remove(element)
                break

    # Convert list in a format similiar to masks
    results = []
    classifications = []
    for element in residual_bboxes:
        mask = np.zeros(shape, dtype=bool)

        bbox = element[0].bounds
        bbox = tuple(map(int, map(np.ceil, bbox)))

        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = True

        classifications.append(element[1])
        results.append({"segmentation": mask, "bbox": bbox})

    return results, classifications


def create_mask_information(sam_masks, classifications, id2label):
    masks = []
    for sam_mask, classification in zip(sam_masks, classifications):
        masks.append(DetectedMask(sam_mask, classification, id2label))
    return masks


def zeroout_mask(image, mask):
    assert image.shape == mask.shape
    mask = dilation(mask, square(3))

    image[mask] = 0
    return image


def remove_erd_elements(image, elements):
    image_grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    for element in elements:
        # ignore edge information
        if element.classification != "Edge Information":
            mask = element.segmentation
            image_grayscale = zeroout_mask(image_grayscale, mask)

    return image_grayscale


def tailor_image(image, elements):
    leftmost, rightmost, lowest, highest = float("inf"), 0, float("inf"), 0

    for element in elements:
        x, y, width, height = element.bbox

        if x < leftmost:
            leftmost = x
        if x + width > rightmost:
            rightmost = x + width
        if y < lowest:
            lowest = y
        if y + height > highest:
            highest = y + height

    zeroed_image = image.copy()
    zeroed_image[0 : int(lowest), :] = 0
    zeroed_image[int(highest) :, :] = 0
    zeroed_image[:, 0 : int(leftmost)] = 0
    zeroed_image[:, int(rightmost) :] = 0

    return zeroed_image
