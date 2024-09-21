import albumentations as A
import cv2 as cv
import numpy as np


def adaptive_threshold(image, **kwargs):
    grayscale = image[:, :, 0]
    grayscale = cv.GaussianBlur(grayscale, (5, 5), 0)
    binary = cv.adaptiveThreshold(
        grayscale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 21
    )
    expanded = np.expand_dims(binary, axis=2)
    return np.repeat(expanded, 3, axis=2)


transform = A.Compose(
    [
        A.LongestMaxSize(max_size=1280),
        A.PadIfNeeded(
            min_height=1280, min_width=1280, border_mode=cv.BORDER_CONSTANT, value=0
        ),
        A.ToGray(p=1.0),
        A.CLAHE(p=1.0),
        A.Lambda(image=adaptive_threshold),
    ]
)


def yolo_preprocess(img):
    transformed = transform(image=img)
    return transformed["image"]

trocr_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=384, interpolation=cv.INTER_LANCZOS4),
        A.Sharpen(alpha=1.0, lightness=1.0, p=1.0),
        A.PadIfNeeded(
            min_height=384,
            min_width=384,
            border_mode=cv.BORDER_CONSTANT,
            value=255,
        ),
    ]
)


def trocr_preprocessor(img_orig, img_prep):
    ref_size, ref_width, _ = img_prep.shape
    assert ref_size == ref_width
    height, width, _ = img_orig.shape

    # calculate sclaing and offsets
    if height > width:
        scaling = height / ref_size
        offset_x = (ref_size - width / scaling) // 2
        offset_y = 0
    else:
        scaling = width / ref_size
        offset_x = 0
        offset_y = (ref_size - height / scaling) // 2

    def get_image(obj):
        x1, y1, x2, y2 = obj.bbox
        x1 = int(round((x1 - offset_x) * scaling))
        y1 = int(round((y1 - offset_y) * scaling))
        x2 = int(round((x2 - offset_x) * scaling))
        y2 = int(round((y2 - offset_y) * scaling))
        obj.bbox = [x1, y1, x2, y2]

        patch = img_orig[y1:y2, x1:x2]
        transformed = trocr_transform(image=patch)
        return transformed["image"]

    return get_image
