import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
from beerscan.webscraping.belgianbeerfactory import crop_image


def scrape_from_internet(npage=1, start_page=1):
    ''' Use `requests` to get the HTML pages'''
    responses = []
    for i in range(npage):
        page = start_page + i
        print(f'Fetching page {page}')
        response = requests.get(f"https://www.saveur-biere.com/fr/3-bouteilles/p-{page}")
        if len(response.history) > 0:
            break
        responses.append(response)
    return responses


def parse(html):
    '''Dummy docstring'''
    soup = BeautifulSoup(html, "html.parser")
    beers = []
    for beer in soup.find_all("div", class_="Box-sc-1duthk5-0 dcljmc"):
        beer_name = beer.find("a", class_="Box-sc-1duthk5-0 ddfLdg").string.strip()
        beer_name = beer_name.replace('/', '-').replace('.', '_').lower()
        # beer_name_list = beer_name.split()
        # volume  = beer_name_list[-1]
        # beer_name = " ".join(beer_name_list[:-1]) + f" - {volume}"


        # beer_name = " ".join(beer_name.replace('/', '-').split()[:-1])
        image_url = beer.find("img", class_='styled__ProductImage-sc-1l219wl-0 gSCQFq Box-sc-1duthk5-0 dxJhnc').get('src', 'Sorry')

        extension = "webp"
        beer_name_img = "_".join(beer_name.replace('-', ' ').split())
        image_name = f'sb_{beer_name_img}.{extension}'

        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            image_path = f"raw_data/images/{image_name}"
            urllib.request.urlretrieve(image_url, image_path)
            if crop_image(image_path):
                beers.append({"beer_name":beer_name, "image_path": image_name})
            else:
                os.remove(image_path)
        else:
            print(f"Error for {beer_name}")
    return beers


def main_sb():
    beers = []
    for i, page in enumerate(scrape_from_internet(11)):
        print(f'Scraping page {i+1}')
        beers += (parse(page.content))
    df = pd.DataFrame(beers)
    print(df)
    df.to_csv('raw_data/csv/sb_scraping.csv')


if __name__ == '__main__':
    main_sb()
