import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import cv2
import  numpy as np

def crop_image(image_path):

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    prototxt = 'beerscan/model/bottle_detection/MobileNetSSD_deploy.prototxt.txt'
    caffe_model = 'beerscan/model/bottle_detection/MobileNetSSD_deploy.caffemodel'

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	(300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()


    found_bottle = False

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        is_bottle = CLASSES[idx] == 'bottle'
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if is_bottle and confidence > 0.85:

            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = np.maximum(np.array(box), 0)
            (startX, startY, endX, endY) = box.astype("int")
            img_crop = image[startY:endY, startX:endX, :]

            cv2.imwrite(image_path, img_crop)
            found_bottle = True

    return found_bottle


def scrape_from_internet(npage=1, start_page=1):
    ''' Use `requests` to get the HTML pages'''
    responses = []
    for i in range(npage):
        page = start_page + i
        print(f'Fetching page {page}')
        response = requests.get(f"https://www.belgianbeerfactory.com/fr/biere-belge/page{page}.html")
        if len(response.history) > 0:
            break
        responses.append(response)
    return responses


def parse(html):
    '''Dummy docstring'''
    soup = BeautifulSoup(html, "html.parser")
    beers = []
    for beer in soup.find_all("div", class_ = "product-block text-left"):
        beer_name = beer.find("a", class_ = "title").string.strip()
        beer_name = beer_name.replace('/', '-').replace('.', '_').lower()
        image_url = beer.find("img").get('src', False)
        if not image_url:
            image_url = beer.find("img").get('data-src', 'Sorry')
        extension = image_url.split('.')[-1]
        beer_name_img = "_".join(beer_name.replace('-', ' ').split())
        image_name = f'bbf_{beer_name_img}.{extension}'

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


def main_bbf():
    beers = []
    for i, page in enumerate(scrape_from_internet(51)):
        print(f'Scraping page {i+1}')
        beers += (parse(page.content))
    df = pd.DataFrame(beers)
    print(df)
    df.to_csv('raw_data/csv/bbf_scraping.csv')


if __name__ == '__main__':
    main_bbf()
