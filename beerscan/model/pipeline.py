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
from beerscan.model.beer_identification.sift import load_sift_dataset, do_sift, identify
import cv2
from beer_identification.image_enhance import contrast

import matplotlib.pyplot as plt

# For pipe testing
from beerscan.api.query import test_boxes_endpoint
import base64

NUM_FEATURES = 300


def main_pipe(image_b64) -> dict:

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
    boxes = test_boxes_endpoint(image_b64)
    image = base64.b64decode(image_b64)
    image = cv2.imdecode(image, flags=1)

    for key, box in boxes.items():
        #Crop Based on box
        (startX, startY, endX, endY) = box["startX"], box["startY"], box["endX"], box["endY"]
        image_cropped = image[startY:endY, startX:endX, :]

        # Contrast cropped image
        image_contrasted = contrast(image_cropped)

        # SIFT cropped image
        keypoints, descriptors = do_sift(image_contrasted, NUM_FEATURES)

        # Identify cropped image
        sift_dataset = load_sift_dataset()
        identification = identify(descriptors, sift_dataset, number=1)["beer_name"]
        boxes[key]["beer_name"] = [name for name in identification]

    return boxes

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
    boxes = main_pipe(data, image)

    # Draw rectangles and print image
    for key, box in boxes["boxes"].items():
        cv2.rectangle(image, (box["startX"], box["startY"]), (box["endX"], box["endY"]), (255,0,0), 2)
        cv2.putText(image, box["beer_name"][0], (box["startX"], box["startY"]-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)
