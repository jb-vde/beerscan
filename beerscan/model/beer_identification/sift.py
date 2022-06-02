import numpy as np
import pandas as pd
import cv2 as cv
from scipy import ndimage, misc
import os

from sklearn.neighbors import NearestNeighbors

from annoy import AnnoyIndex

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

    start_time = time.time()

    n_features = 300                        # Features to extract from images
    image_directory = "raw_data/images/bbf/" # Dataset directory
    images_df = pd.read_csv("raw_data/csv/bbf_scraping.csv") # CSV describing dataset
    IMG = cv.imread('raw_data/images/jupi.jpg') # Image to identify

    ###################
    # SIFT on dataset #
    ###################

    knn_mapping = []
    all_descriptors = []

    print("> SIFT Dataset")

    for index, row in images_df.iterrows():
        img = cv.imread(os.path.join(image_directory, row["image_path"]))
        kp, desc = do_sift(img, n_features)
        for vector in desc:
            all_descriptors.append(vector)
            # Match index of vector with corresponding beer name
            knn_mapping.append(row["beer_name"])

    print("--- %s seconds ---" % (time.time() - start_time))

    #############################
    # SIFT on image to identify #
    #############################

    keypoints, descriptors = do_sift(IMG, n_features)

    ###########################
    # KNN to compare features #
    ###########################

    knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
    knn.fit(all_descriptors)

    images_df["score"] = 0

    print("> KNN Descriptors")

    counter = 0
    for vect in descriptors:
        print(f"computing {counter}/{n_features}")
        neighbors = knn.kneighbors([vect], return_distance=False)[0]
        print(neighbors)
        for i in range(len(neighbors)):
            print(knn_mapping[neighbors[i]], neighbors[i], f"score += {(len(neighbors) - i) * 2}")
            images_df.loc[images_df['beer_name'] == knn_mapping[neighbors[i]], 'score'] += (len(neighbors) - i) * 2
        counter+=1

    print(images_df.sort_values(by=["score"], ascending=False).head(5))
    print("--- %s seconds ---" % (time.time() - start_time))
