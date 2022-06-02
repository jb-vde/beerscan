import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import re

def scrape_images(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')

    beers = soup.find_all("article")

    images = []
    names = []

    for beer in beers:
        image = beer.find("img", {"class": "product-miniature__img"})["src"]
        beer_name = beer.find("h2", {"class": "product-miniature__title"}).text.strip().lower()
        volume = beer.find(class_= "product-miniature__feature", string=re.compile("[0-9]cl"))
        if image and beer_name and volume:
            names.append(f"{beer_name} - {volume.text.strip()}")
            images.append(image)

    return pd.DataFrame({"beer_name": names,
                         "image_url": images})


def download_images(scrape_results, path):
    image_path = []
    images = scrape_results["image_url"]
    names = scrape_results["beer_name"]
    for image_url, beer_name in zip(images, names):
        extension = "." + image_url.split(".")[-1]
        image_name = beer_name.replace(".", " ")\
                              .replace("/", " ")\
                              .replace("-", " ")\
                              .replace("\\", " ") + extension
        image_name = "_".join(image_name.split())
        try:
            urlretrieve(image_url, path + image_name)
            image_path.append(image_name)
        except:
            print(beer_name, image_url)
            names = names[names != beer_name]
            images = images[images != image_url]
            print(f"\t> Couldn't download {image_url}")

    return pd.DataFrame({"name": names,
                         "image_path": image_path,
                         "image_url": images})


def scrape_next_page(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')

    next_page = soup.find("a", {"rel": "next", "class": "next"})
    if next_page:
        return next_page["href"]
    return None


if __name__ == "__main__":
    next = "https://biere-speciale.be/fr/3-les-bieres"
    data = pd.DataFrame({"name":[], "image_path":[]})

    while next:
        print("scraping ", next, "...")
        dl_info = download_images(scrape_images(next), "raw_data/images/bs/")
        data = pd.concat([data, dl_info.drop(columns=["image_url"])], ignore_index=True)
        next = scrape_next_page(next)

    data.to_csv("raw_data/csv/bs_scraping.csv")
