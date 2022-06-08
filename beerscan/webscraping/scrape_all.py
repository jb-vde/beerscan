
import pandas as pd
from beerscan.webscraping.belgianbeerfactory import main_bbf, crop_image
from beerscan.webscraping.belgianhappiness import main_bh
from beerscan.webscraping.mybeerbox import main_mbb
from beerscan.webscraping.bierespeciale import main_bs


def crop_manual_images():
    manual_df = pd.read_csv("raw_data/csv/manual_import.csv", index_col=0)
    print(manual_df.head())
    for img in manual_df['image_path']:
        img_path = f"raw_data/images/{img}"
        print(img_path)
        crop_image(img_path)


def main():
    print("Scraping website: Belgian Beer Factory")
    main_bbf()

    print("Scraping website: Belgian Happiness")
    main_bh()

    print("Scraping website: My Beer Box")
    main_mbb()

    print("Scraping website: Bières spéciales")
    main_bs()

if __name__ == "__main__":
    #crop_manual_images()
    main()
