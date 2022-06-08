from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def flatten(list):

    return [x for sublist in list for x in sublist]


def remove_volume_info(beer_name):

    name_list = beer_name.split(' - ')

    if len(name_list) > 1 and 'cl' in name_list[-1]:
        beer_name = ' - '.join(name_list[:-1])
    else:
        beer_name = ' - '.join(name_list)

    return beer_name


def make_query(beer_name):

    beer_name = remove_volume_info(beer_name)

    query = "+".join(beer_name.split())
    query = "\\'".join(query.split("'"))

    return query


def api_response(beer_info_list):

    key_list = ["brewery", "beer", "style", "overall_score",
                "style_score", "star_rating", "n_reviews"]
    value_list = [el for el in beer_info_list if not el.startswith("Available")]
    #value_list = [el.split('•') for el in value_list]
    value_list = flatten(value_list)
    print("I went here")
    print(value_list)

    return dict(zip(key_list, value_list))


def load_driver():
    opts = Options()
    serv = Service(ChromeDriverManager().install())

    opts.add_argument("-remote-debugging-port=9224")
    opts.add_argument("-headless")
    opts.add_argument("-disable-gpu")
    opts.add_argument("-no-sandbox")

    driver = webdriver.Chrome(service=serv, options=opts)

    return driver


def search_beer(beer_name):

    driver = load_driver()
    print(beer_name)
    query = make_query(beer_name)
    driver.get(f'https://www.ratebeer.com/search?q={query}&tab=beer')

    try:
        beer = WebDriverWait(driver, 100).until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div[class='fg-1']")))
        beer_text = beer.get_attribute('innerText')
        beer_info_list = beer_text.split('\n')

    except TimeoutException:
        print("Loading took too much time!")
        beer_info_list = []



    driver.quit()

    return api_response(beer_info_list)


def main():

    beer_name = "guinness - 33cl"
    response = search_beer(beer_name)

    print(response)


if __name__ == "__main__":
    main()
