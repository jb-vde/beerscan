import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


def scrape_images(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')

    images = soup.find_all("img", {"class": "product-miniature__img"})
    names = soup.find_all("p", {"class": "product-miniature__manufacturer-name"})

    names = [image.text.lower() for image in images]
    images = [image["src"] for image in images]

    return pd.DataFrame({"beer_name": names,
                         "image_url": images})


def download_images(scrape_results, path):
    image_path = []
    images = scrape_results["image_url"]
    names = scrape_results["beer_name"]
    for image_url, beer_name in zip(images, names):
        extension = "." + image_url.split(".")[-1]
        image_name = beer_name.replace(" ", "_")\
                              .replace(".", "")\
                              .replace("/", "")\
                              .replace("\\", "") + extension
        urlretrieve(image_url, path + image_name)
        image_path.append(image_name)

    return pd.DataFrame({"name": names,
                         "image_path": image_path,
                         "image_url": images})


def scrape_next_page(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')

    next_page = soup.find("a", {"rel": "next", "class": "next"})
    return next_page["href"]


if __name__ == "__main__":
    next = scrape_next_page("https://biere-speciale.be/fr/3-les-bieres")
    data = pd.DataFrame({"name":[], "image_path":[]})

    while next:
        print("scraping ", next, "...")
        dl_info = download_images(scrape_images(next), "raw_data/images/")
        data = pd.concat([data, dl_info.drop(columns=["image_url"])], ignore_index=True)
        next = scrape_next_page(next)

    #data.to_csv("raw_data/csv/bs_scraping.csv")
    print(data)
