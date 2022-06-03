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
        response = requests.get(f"https://mybeerbox.be/collections/toutes-les-bieres?page={page}")
        if len(response.history) > 0:
            break
        responses.append(response)
    return responses


def parse(html):
    '''Dummy docstring'''
    soup = BeautifulSoup(html, "html.parser")
    beers = []
    for beer in soup.find_all("div", class_ = "o-layout__item u-1/1 u-1/3@tab u-1/4@desk")[1:]:
        #print(f'parsing beer {i}')

        beer_name = beer.find("h2", class_='product-card__title h4').string.split(' - ')[0].strip()
        beer_name = beer_name.replace('/', '-').replace('.', '_').lower()
        image_url = beer.find("img", class_='product-card__img').get('src', 'Sorry').split('?')[0]
        image_url = f'https:{image_url}'
        extension = image_url.split('.')[-1]
        beer_name_img = "_".join(beer_name.replace('-', ' ').split())
        image_name = f'mbb_{beer_name_img}.{extension}'

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


def main_mbb():
    beers = []
    for i, page in enumerate(scrape_from_internet(19)):
        print(f'Scraping page {i+1}')
        beers += (parse(page.content))
    df = pd.DataFrame(beers)
    print(df)
    df.to_csv('raw_data/csv/mbb_scraping.csv')


if __name__ == '__main__':
    main_mbb()
