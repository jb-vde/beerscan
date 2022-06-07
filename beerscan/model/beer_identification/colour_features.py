import cv2
import numpy as np
import pandas as pd
import os


# Not great, no need to use this module in current state
# Could investigate cropping out the background to keep only the bottle
# Could investigate cropping only the label instead of the entire bottle   <=


# https://stackoverflow.com/questions/67323056/histogram-comparison-between-two-images

def compare_images(img1_path, img2_path):
    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Calculate the histograms, and normalize them
    hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Find the metric value
    return cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)


if __name__ == "__main__":
    IMG1 = "raw_data/images/test_img/cropped_img.jpg"
    IMG2 = "raw_data/images/bbf_duvel_tripel_hop_citra_33cl.jpg"

    verbose = True
    image_directory = "raw_data/images/"


    rank = []

    images_df = pd.read_csv("raw_data/csv/bbf_scraping.csv", index_col=0) # CSV describing dataset
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/bh_scraping.csv", index_col=0)])
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/bs_scraping.csv", index_col=0)])
    images_df = pd.concat([images_df, pd.read_csv("raw_data/csv/mbb_scraping.csv", index_col=0)])

    # Cleaning packs and "lots"
    print(f'Dataset shape before cleaning : {images_df.shape}')
    images_df = images_df[images_df["beer_name"].str.lower().str.contains("pack") == False]
    images_df = images_df[images_df["beer_name"].str.lower().str.contains(" lot ") == False]
    print(f'Cleaned dataset shape : {images_df.shape}')


    for index, row in images_df.iterrows():

            if verbose:
                print(f"\rcomputing {index}", end="")

            img = os.path.join(image_directory, row["image_path"])
            rank.append( np.abs(compare_images(IMG1, img)) )

    print(np.sort(rank, )[:5])
