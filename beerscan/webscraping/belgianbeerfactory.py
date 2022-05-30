import requests
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd


def scrape_from_internet(npage=1, start_page=1):
    ''' Use `requests` to get the HTML pages'''
    responses = []
    for i in range(npage):
        page = start_page + i
        response = requests.get(f"https://www.belgianbeerfactory.com/en/belgian-beer/page{page}.html")
        if len(response.history) > 0:
            break
        responses.append(response)
    return responses


def parse(html):
    '''Dummy docstring'''
    soup = BeautifulSoup(html, "html.parser")
    beers = []
    for beer in soup.find_all("div", class_ = "product-block text-left"):
        beer_name = beer.find("a", class_ = "title").string.split('-')[0].strip()
        beer_name = beer_name.replace('/', '-')
        image_url = beer.find("img").get('src', False)
        if not image_url:
            image_url = beer.find("img").get('data-src', 'Sorry')
        extension = image_url.split('.')[-1]
        image_path = f'bbf_{"_".join(beer_name.split()).lower()}.{extension}'

        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            urllib.request.urlretrieve(image_url, f"raw_data/images/{image_path}")
            beers.append({"beer_name":beer_name, "image_path": image_path})
        else:
            print(f"Error for {beer_name}")
    return beers


def main():
    beers = []
    for i, page in enumerate(scrape_from_internet(51)):
        print(f'Scraping page {i+1}')
        beers += (parse(page.content))
    df = pd.DataFrame(beers)
    print(df)
    df.to_csv('raw_data/csv/bbf_scraping.csv')


if __name__ == '__main__':
    main()
