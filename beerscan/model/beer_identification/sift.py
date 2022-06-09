# Data Structures
import numpy as np
import pandas as pd

# Image Manipulation
import cv2
from beerscan.model.beer_identification.image_enhance import contrast
from beerscan.webscraping.belgianbeerfactory import crop_image

# OS Path Manipluation
import os

# Model Estimator
from beerscan.model.estimators import annoy_est

# For testing and debugging
import time

# Saved Models
ANN_MODEL_PATH = "beerscan/data/model.ann"
SIFT_DATASET_PATH = "beerscan/data/dataset_sift.csv"


def do_sift(img:list, features:int) -> tuple:
    """
    Extracts image features based on the SIFT algorithm
        Parameters:
            img      (list) : array representing an image
            features (int) : number of features to extract
        Returns:
            (tuple): (keypoints, descriptors)
    """

    # SIFT doesn't care about colors and gray is faster than colours
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=features)
    keypoints, descriptors = sift.detectAndCompute(gray,None)

    # Bug in sift function, sometimes returns n+1 features
    # https://stackoverflow.com/questions/56729238/how-to-fix-the-number-of-sift-keypoints
    return (keypoints[:features], descriptors[:features])


def draw_keypoints(img, keypoints):
    # Not used currently
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.drawKeypoints(gray,keypoints,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def sift_from_dataframe(images_df:pd.DataFrame, verbose:bool=False) -> pd.DataFrame:
    """

    """
    beer_names = []
    keypoints = []
    descriptors = []

    if verbose:
        print("> SIFT Dataset")
        start_time = time.time()

    # TODO: Check images_df format (columns match expectations)

    for index, row in images_df.iterrows():

        if verbose:
            print(f"\rcomputing {index}", end="")

        img = cv2.imread(os.path.join(image_directory, row["image_path"]))

        img = contrast(img)

        kp, desc = do_sift(img, n_features)
        for vector, keyp in zip(desc, kp):
            descriptors.append(vector)
            keypoints.append(keyp)
            beer_names.append(row["beer_name"])

    if verbose:
        run_time = time.time() - start_time
        print(f"\nSIFT ran in {run_time} seconds")

    return pd.DataFrame({"beer_name" : beer_names,
                            "keypoint" : keypoints,
                            "descriptor" : descriptors})


def load_sift_dataset(images_df=False):
    try:
        dataset_sift = pd.read_csv(SIFT_DATASET_PATH)
    except IOError:
        dataset_sift = sift_from_dataframe(images_df, verbose=True)
        dataset_sift.to_csv(SIFT_DATASET_PATH)
    return dataset_sift



def identify(descriptors:list, sift_dataset:pd.DataFrame, number:int=1):

    vec_dim = len(descriptors[0])

    df = pd.DataFrame()
    df["beer_name"] = list(set(sift_dataset['beer_name']))
    df["score"] = 0

    try:
        annoy = annoy_est.load_annoy(vec_dim, ANN_MODEL_PATH)
    except IOError:
        annoy = annoy_est.build_from_df(vec_dim, sift_dataset)
        annoy.save(ANN_MODEL_PATH)

    neighbors = annoy_est.run_annoy(descriptors, annoy, verbose=False)

    for group in neighbors:
        for i in range(len(group)):
            df.loc[df["beer_name"] == sift_dataset.at[group[i], "beer_name"], 'score'] += (len(group) - i)**2

    return df.sort_values(by=["score"], ascending=False).head(5)[:number]



if __name__ == "__main__":

    DRAW_FEATURES = True

    start_time = time.time()

    n_features = 200  # Number of features to extract from images
    image_directory = "raw_data/images/" # Dataset directory

    images_df = pd.read_csv("raw_data/csv/bbf_scraping.csv", index_col=0) # CSV describing dataset
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/bh_scraping.csv", index_col=0)])
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/bs_scraping.csv", index_col=0)])
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/mbb_scraping.csv", index_col=0)])
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/manual_import.csv", index_col=0)])

    IMG_PATH = 'raw_data/images/test_img/belgian_beer_tour.jpg'
    crop_image(IMG_PATH)
    IMG = cv2.imread(IMG_PATH) # Image to identify


    # Cleaning packs and "lots"
    print(f'Dataset shape before cleaning : {images_df.shape}')
    images_df = images_df[images_df["beer_name"].str.lower().str.contains("pack") == False]
    images_df = images_df[images_df["beer_name"].str.lower().str.contains(" lot ") == False]
    print(f'Cleaned dataset shape : {images_df.shape}')

    ###################
    # SIFT on dataset #
    ###################


    dataset_sift = load_sift_dataset(images_df)

    print(dataset_sift)

    #############################
    # SIFT on image to identify #
    #############################

    keypoints, descriptors = do_sift(IMG, n_features)

    images_df["score"] = 0   # Scoring for similarity comparison

    #########################################
    #                 ANNOY                 #
    # Approximate Nearest Neighbors Oh Yeah #
    #    https://github.com/spotify/annoy   #
    #########################################

    print("\n",identify(descriptors, dataset_sift, number = 3))
    print("--- %s seconds ---" % (time.time() - start_time))

    # TODO: Draw features on images
