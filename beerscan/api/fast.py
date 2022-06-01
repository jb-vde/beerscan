from sys import flags
from fastapi import FastAPI, File, Request
import cv2
import numpy as np
import  base64

app = FastAPI()

@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.post("/size")
def check_size(file: bytes = File(...)):
    #Check type of image as it arrives.
    print(
         f'''
         --------------------------------------------------------------\n
         --------------------------------------------------------------\n
         \n\n\nAfter being passed to API, file has type: {type(file)}\n
         The file has length {len(file)}
         '''
         )
    # convert to bytes with bytearray, and to np array
    decoded_file=base64.decodebytes(file)

    return f'This file is {len(decoded_file)/1000} Kbytes and type {type(decoded_file)}'


@app.post("/predict_boxes")
async def predict_boxes(request: Request):

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    prototxt = 'beerscan/model/bottle_detection/MobileNetSSD_deploy.prototxt.txt'
    caffe_model = 'beerscan/model/bottle_detection/MobileNetSSD_deploy.caffemodel'

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

    request_body = await request.json()
    image = base64.b64decode(request_body['image'])
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, flags=1)

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


        if is_bottle and confidence > 0.5:
            j += 1
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = np.maximum(np.array(box), 0).astype(int).tolist()
            keys = ['startX', 'startY', 'endX', 'endY']
            dic = dict(zip(keys, box))
            boxes[f"box_{j}"] = dic

    return {'boxes': boxes}
