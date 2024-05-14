import cv2 as cv
import numpy as np


def get_bboxes(imgs, mdl):
    all_bboxes_cls = []
    results = []
    for img in imgs:
        results = mdl.predict(img, imgsz=1280, conf=0.2, iou=0.5, half=True)

        bboxes = results[0].boxes.xyxy
        bboxes = bboxes.cpu().numpy()

        confidence = results[0].boxes.conf
        confidence = confidence.cpu().numpy()

        classes = results[0].boxes.cls
        classes = classes.cpu().numpy()

        triples = list(zip(bboxes, confidence, classes))
        all_bboxes_cls.append(triples)

    cls_names = results[0].names
    return all_bboxes_cls, cls_names


def cut_boxes(all_imgs, all_bboxes_cls, ignore):
    cut_imgs = []
    for img, bboxes_cls in zip(all_imgs, all_bboxes_cls):
        mask = np.zeros_like(img, dtype=np.uint8)
        for bbox_cl in bboxes_cls:
            if bbox_cl[2] in ignore:
                continue
            x1, y1, x2, y2 = bbox_cl[0]
            a = (int(x1.item()), int(y1.item()))
            b = (int(x2.item()), int(y2.item()))
            cv.rectangle(mask, a, b, (255, 255, 255), -1)
            mask = cv.bitwise_and(img, mask)

        cut_imgs.append(mask)
    return cut_imgs
