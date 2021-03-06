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
        response = requests.get(f"https://www.belgianhappiness.com/en/beers?page={page}")
        if len(response.history) > 0:
            break
        responses.append(response)
    return responses


def parse(html):
    '''Dummy docstring'''
    soup = BeautifulSoup(html, "html.parser")
    beers = []
    for beer in soup.find_all("div", class_="product-item"):
        beer_name = beer.find("span", class_="title").string.split(' - ')[0].strip()
        beer_name = beer_name.replace('/', '-').replace('.', '_').lower()
        beer_name_list = beer_name.split()
        volume  = beer_name_list[-1]
        beer_name = " ".join(beer_name_list[:-1]) + f" - {volume}"


        # beer_name = " ".join(beer_name.replace('/', '-').split()[:-1])
        image_url = beer.find("img", class_='img-responsive').get('src', 'Sorry')

        extension = image_url.split('.')[-1]
        beer_name_img = "_".join(beer_name.replace('-', ' ').split())
        image_name = f'bh_{beer_name_img}.{extension}'

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


def main_bh():
    beers = []
    for i, page in enumerate(scrape_from_internet(29)):
        print(f'Scraping page {i+1}')
        beers += (parse(page.content))
    df = pd.DataFrame(beers)
    print(df)
    df.to_csv('raw_data/csv/bh_scraping.csv')


if __name__ == '__main__':
    main_bh()
