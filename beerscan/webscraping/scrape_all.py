
from beerscan.webscraping.belgianbeerfactory import main_bbf
from beerscan.webscraping.belgianhappiness import main_bh
from beerscan.webscraping.mybeerbox import main_mbb
from beerscan.webscraping.bierespeciale import main_bs


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
    main()
