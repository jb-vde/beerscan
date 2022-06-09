# Image Manipulation
import cv2
from beerscan.model.beer_identification.image_enhance import contrast

# Bottle Detection
from beerscan.model.bottle_detection.mobilenet_ssd import detect_bottles

# Beer Identification
from beerscan.model.beer_identification.sift import do_sift

# SIFT parameters
NUM_FEATURES = 300


# Thanks https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def get_steps(img, output_dir):
    # Get boxes
    data = detect_bottles(img)
    # TODO:Draw boxes and save image
    for key, box in data.items():
        #Crop Based on box
        (boxSX, boxSY, boxEX, boxEY) = box["startX"], box["startY"], box["endX"], box["endY"]
        img_detection = img.copy()
        cv2.rectangle(img_detection, (boxSX, boxSY), (boxEX, boxEY), (255,0,0), 2)
        cv2.imwrite(output_dir + "detection.jpg", img_detection)

    for key, box in data.items():
        #Crop Based on box
        (startX, startY, endX, endY) = box["startX"], box["startY"], box["endX"], box["endY"]
        image_cropped = img[startY:endY, startX:endX, :]
        # TODO: save image cropped
        cv2.imwrite(output_dir + "cropping.jpg", image_cropped)

        # Contrast cropped image
        image_contrasted = contrast(image_cropped)
        # TODO: save image contrasted
        cv2.imwrite(output_dir + "contrasting.jpg", image_contrasted)

        # SIFT cropped image
        keypoints, descriptors = do_sift(image_contrasted, NUM_FEATURES)
        print(keypoints)
        # Draw keypoints and save image
        img_kp = cv2.drawKeypoints(image_contrasted, keypoints, 0,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #image_keypoints = draw_keypoints(image_contrasted, keypoints)
        cv2.imwrite(output_dir + "keypoints.jpg", img_kp)

if __name__ == "__main__":


    IMG_PATH = 'raw_data/images/test_img/jupi_mael.jpg'
    IMG = cv2.imread(IMG_PATH)

    get_steps(IMG, "raw_data/output/")
