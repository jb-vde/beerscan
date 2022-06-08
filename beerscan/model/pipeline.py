# Image Manipulation
import cv2
from beerscan.model.beer_identification.image_enhance import contrast

# Bottle Detection
from beerscan.model.bottle_detection.mobilenet_ssd import detect_bottles

# Beer Identification
from beerscan.model.beer_identification.sift import load_sift_dataset, do_sift, identify
from beerscan.api.ratebeer_api import search_beer


# SIFT Parameters
NUM_FEATURES = 300


def main_pipe(image:list) -> dict:
    """
    Full pipeline to identify beers from a given image
        Parameters:
            image (list): an array representing an image
        Returns:
            data (dict): dictionnary with following keys:
                startX (int) - X coordinate of top left corner\n
                startY (int) - Y coordinate of top left corner\n
                endX   (int) - X coordinate of bottom right corner\n
                endY   (int) - Y coordinate of bottom left corner\n
                beer_name (str) - Name of the beer (raw)\n
                info   (dict) - {"brewery", "beer", "style", "abv", "overall_score",
                                "style_score", "star_rating", "n_reviews"}
    """
    # Get boxes
    data = detect_bottles(image)

    # Load the sift dataset
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
        #identification = identification[identification["score"] > 70]["beer_name"]

        data[key]["beer_name"] = [name for name in identification]
        data[key]["info"] = search_beer(identification.iloc[0])

    return data


if __name__ == "__main__":

    image_file = 'raw_data/images/test_img/belgian_beer_tour.jpg'
    image = cv2.imread(image_file)

    # Test the pipe
    data = main_pipe(image)

    print(data)
