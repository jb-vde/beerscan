import cv2
import numpy as np

# Model parameters
PROTOTXT = 'beerscan/model/bottle_detection/MobileNetSSD_deploy.prototxt.txt'
CAFFE_MODEL = 'beerscan/model/bottle_detection/MobileNetSSD_deploy.caffemodel'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]


def detect_bottles(image):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	(300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    boxes = {}
    # loop over the detections
    j = 0
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        is_bottle = CLASSES[idx] == 'bottle'
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if is_bottle and confidence > 0.8:
            j += 1
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = np.maximum(np.array(box), 0).astype(int).tolist()
            keys = ['startX', 'startY', 'endX', 'endY']
            dic = dict(zip(keys, box))
            boxes[f"beer_{j}"] = dic

    return boxes
