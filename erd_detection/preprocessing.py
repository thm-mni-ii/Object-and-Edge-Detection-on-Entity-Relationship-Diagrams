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
