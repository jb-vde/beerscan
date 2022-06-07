# Image Manipulation
import cv2
# Image Preprocessing
from beer_identification.image_enhance import contrast

# Bottle Detection
from beerscan.model.bottle_detection.mobilenet_ssd import detect_bottles

# Beer Identification
from beerscan.model.beer_identification.sift import load_sift_dataset, do_sift, identify
from beerscan.api.ratebeer_api import search_beer

# Bit Manipulation - for pipe testing
import base64


# SIFT Parameters
NUM_FEATURES = 300


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

        data[key]["beer_name"] = [name for name in identification]
        data[key]["info"] = search_beer(identification.iloc[0])

    return data


if __name__ == "__main__":

    image_file = 'raw_data/images/test_img/belgian_beer_tour.jpg'
    image = cv2.imread(image_file)

    # Test the pipe
    data = main_pipe(image)

    print(data)
