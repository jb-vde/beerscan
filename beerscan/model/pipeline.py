"""
    Input JSON with images and boxes
    Crop image based on boxes
    Contrast cropped img
    SIFT cropped img
    Indentify by comparing features against dataset
    return JSON with beer name
"""


"""
    IN :

    img_dict = {
        box1: {
            'startX',
            'startY',
            'endX',
            'endY'
        },
        box2: {
            'startX',
            'startY',
            'endX',
            'endY'
        },
        ...
        boxN: {
            'startX',
            'startY',
            'endX',
            'endY'
        }
    }

"""
from sys import api_version
from pandas import DataFrame
from beerscan.api.ratebeer_api import search_beer
from beerscan.model.beer_identification.sift import load_sift_dataset, do_sift, identify
from beerscan.model.bottle_detection.mobilenet_ssd import detect_bottles
import cv2
from beer_identification.image_enhance import contrast

import matplotlib.pyplot as plt
import numpy as np

# For pipe testing
from beerscan.api.query import test_boxes_endpoint
import base64

NUM_FEATURES = 300

def img_from_b64(image_b64):
    image = base64.b64decode(image_b64)
    image = np.frombuffer(image, dtype=np.uint8)
    return cv2.imdecode(image, flags=1)


def main_pipe(image) -> dict:

    """
    IN => IMG in bytes
         v IMG_bytes => get boxes
         v IMG to array
         v for each box:
         v   crop IMG_array according to box
         v   Contrast cropped_IMG
         v   Get features
         v   Identify image
            Retrieve beer info based on clean name
            Add name, informations and boxes to dict
    OUT =>return dict
    """
    data = detect_bottles(image)

    sift_dataset = load_sift_dataset()

    for key, box in data.items():
        print(box)
        #Crop Based on box
        (startX, startY, endX, endY) = box["startX"], box["startY"], box["endX"], box["endY"]
        image_cropped = image[startY:endY, startX:endX, :]

        # Contrast cropped image
        image_contrasted = contrast(image_cropped)

        # SIFT cropped image
        keypoints, descriptors = do_sift(image_contrasted, NUM_FEATURES)

        # Identify cropped image
        identification = identify(descriptors, sift_dataset, number=1)["beer_name"]
        print(identification)

        data[key]["beer_name"] = [name for name in identification]
        #data[key]["info"] = search_beer(identification.iloc[0])
        data[key]["info"] = {}

    return data

"""
OUT:

    img_dict = {
            box1: {
                'startX',
                'startY',
                'endX',
                'endY',
                'beer_name',
                'info'
            },
            box2: {
                'startX',
                'startY',
                'endX',
                'endY',
                'beer_name'
            },
            ...
            boxN: {
                'startX',
                'startY',
                'endX',
                'endY',
                'beer_name'
            }
        }
    }

"""

if __name__ == "__main__":

    # Get the boxes

    image_file = 'raw_data/images/test_img/belgian_beer_tour.jpg'
    image = cv2.imread(image_file)

    with open(image_file, "rb") as f:
        im_bytes = f.read()
    img_b64 = base64.b64encode(im_bytes).decode("utf8")

    # Test the pipe
    data = main_pipe(img_b64)

    print(data)
