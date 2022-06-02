import numpy as np
import pandas as pd
import cv2 as cv
from scipy import ndimage, misc
import os


from sklearn.neighbors import NearestNeighbors

from beerscan.model.estimators import annoy_est


import time


def do_sift(img, features):

    # SIFT doesn't care about colors and gray is faster than colours
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=features)
    keypoints, descriptors = sift.detectAndCompute(gray,None)

    # Bug in sift function, sometimes returns n+1 features
    # https://stackoverflow.com/questions/56729238/how-to-fix-the-number-of-sift-keypoints
    return (keypoints[:features], descriptors[:features])


def draw_keypoints(img, keypoints):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return cv.drawKeypoints(gray,keypoints,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



if __name__ == "__main__":

    ANNOY = True

    start_time = time.time()

    n_features = 300  # Number of features to extract from images
    image_directory = "raw_data/images/bbf/" # Dataset directory
    images_df = pd.read_csv("raw_data/csv/bbf_scraping.csv") # CSV describing dataset
    IMG = cv.imread('raw_data/images/jupi.jpg') # Image to identify

    # Cleaning packs and "lots"
    print(f'Dataset shape before cleaning : {images_df.shape}')
    images_df = images_df[images_df["beer_name"].str.lower().str.contains("pack") == False]
    images_df = images_df[images_df["beer_name"].str.lower().str.contains(" lot ") == False]
    print(f'Cleaned dataset shape : {images_df.shape}')

    ###################
    # SIFT on dataset #
    ###################

    mapping = []
    all_descriptors = []
    all_keypoints = []

    print("> SIFT Dataset")

    for index, row in images_df.iterrows():
        print(f"\rcomputing {index}", end="")
        img = cv.imread(os.path.join(image_directory, row["image_path"]))
        kp, desc = do_sift(img, n_features)
        for vector, keyp in zip(desc, kp):
            all_descriptors.append(vector)
            all_keypoints.append(keyp)
            # Match index of vector with corresponding beer name
            mapping.append(row["beer_name"])

    print("\n--- %s seconds ---" % (time.time() - start_time))

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
    if ANNOY:

        neighbors = annoy_est.annoy(descriptors, all_descriptors, metric="angular", trees=10, verbose=1)
        for group in neighbors:
            for i in range(len(group)):
                images_df.loc[images_df['beer_name'] == mapping[group[i]], 'score'] += (len(group) - i)**2

        print("\n",images_df.sort_values(by=["score"], ascending=False).head(5))
        print("--- %s seconds ---" % (time.time() - start_time))


    ###########################
    # KNN to compare features #
    ###########################
    else:
        print("\n####  KNN  ####\n")
        knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        knn.fit(all_descriptors)

        print("> KNN Descriptors")

        counter = 0
        for vect in descriptors:
            print(f"\rcomputing {counter}/{n_features}", end="")
            neighbors = knn.kneighbors([vect], return_distance=False)[0]
            for i in range(len(neighbors)):
                images_df.loc[images_df['beer_name'] == mapping[neighbors[i]], 'score'] += (len(neighbors) - i)**2
            counter+=1

        print("\n",images_df.sort_values(by=["score"], ascending=False).head(5))
        print("--- %s seconds ---" % (time.time() - start_time))
